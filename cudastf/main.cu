#include <cuda/experimental/stf.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "arguments.h"
#include "core.h"
#include "data_access.h"
#include "expanded_dag.h"
#include "identity.h"
#include "results.h"
#include "topology.h"
#include "workload.cuh"

using cuda::experimental::stf::exec_place;
using cuda::experimental::stf::logical_data;
using cuda::experimental::stf::slice;
using cuda::experimental::stf::stream_ctx;

namespace {

using Clock = std::chrono::steady_clock;
using LogicalByteData = logical_data<slice<unsigned char>>;

void cuda_check(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class CudaEvent {
public:
  explicit CudaEvent(const char *create_operation)
  {
    cuda_check(cudaEventCreate(&event_), create_operation);
  }

  ~CudaEvent()
  {
    if (event_ != nullptr) {
      (void)cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent &) = delete;
  CudaEvent &operator=(const CudaEvent &) = delete;

  cudaEvent_t get() const { return event_; }

  void destroy_checked(const char *operation)
  {
    cudaEvent_t event = std::exchange(event_, nullptr);
    if (event != nullptr) {
      cuda_check(cudaEventDestroy(event), operation);
    }
  }

private:
  cudaEvent_t event_ = nullptr;
};

double milliseconds(Clock::time_point start, Clock::time_point stop)
{
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

std::string make_task_symbol(const char *task_type,
                             const TaskCoordinates &coordinates)
{
  return "T(" + std::string(task_type) + ",(" +
         std::to_string(coordinates.dag_index) + "," +
         std::to_string(coordinates.timestep) + "," +
         std::to_string(coordinates.point) + "))";
}

std::string make_data_symbol(std::int64_t dag_index, long field, long point)
{
  return "D(" + std::to_string(dag_index) + "," +
         std::to_string(field) + "," + std::to_string(point) + ")";
}

std::string format_uuid(const cudaUUID_t &uuid)
{
  std::ostringstream out;
  out << "GPU-";
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) out << '-';
    out << std::hex << std::setfill('0') << std::setw(2)
        << (static_cast<unsigned int>(
                static_cast<unsigned char>(uuid.bytes[i])));
  }
  return out.str();
}

CudaDeviceInfo inspect_cuda_device(int device_id)
{
  cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
  cudaDeviceProp properties{};
  cuda_check(cudaGetDeviceProperties(&properties, device_id),
             "cudaGetDeviceProperties");
  CudaDeviceInfo result;
  result.device_id = device_id;
  result.name = properties.name;
  result.uuid = format_uuid(properties.uuid);
  result.compute_capability_major = properties.major;
  result.compute_capability_minor = properties.minor;
  result.max_threads_per_block = properties.maxThreadsPerBlock;
  result.max_grid_size_x = properties.maxGridSize[0];
  result.legacy_shared_memory_per_block = properties.sharedMemPerBlock;
  result.optin_shared_memory_per_block =
      std::max(properties.sharedMemPerBlock,
               properties.sharedMemPerBlockOptin);
  cuda_check(cudaRuntimeGetVersion(&result.runtime_version),
             "cudaRuntimeGetVersion");
  cuda_check(cudaDriverGetVersion(&result.driver_version),
             "cudaDriverGetVersion");
  return result;
}

std::vector<CudaDeviceInfo> inspect_cuda_devices(
    const TaskPlacement &task_placement)
{
  if (task_placement.devices.empty()) {
    throw std::runtime_error("at least one CUDA device is required");
  }
  int count = 0;
  cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  std::set<int> seen;
  std::vector<CudaDeviceInfo> result;
  result.reserve(task_placement.devices.size());
  for (int device_id : task_placement.devices) {
    if (device_id < 0 || device_id >= count) {
      std::ostringstream message;
      message << "invalid CUDA device " << device_id
              << "; available range is [0," << count - 1 << "]";
      throw std::runtime_error(message.str());
    }
    if (!seen.insert(device_id).second) {
      throw std::runtime_error(
          "CUDA device list contains duplicate device " +
          std::to_string(device_id));
    }
    result.push_back(inspect_cuda_device(device_id));
  }
  cuda_check(
      cudaSetDevice(task_placement.devices.front()), "cudaSetDevice");
  return result;
}

struct DependencyDataStore {
  const ExpandedDag *expanded_dag = nullptr;
  std::vector<LogicalByteData> slots;

  LogicalByteData &at(const DataId &data)
  {
    if (data.dag_index != expanded_dag->dag_index()) {
      throw std::logic_error("data id does not belong to this DAG");
    }
    const std::size_t index =
        static_cast<std::size_t>(data.field) *
            expanded_dag->task_graph.max_width +
        static_cast<std::size_t>(data.point);
    return slots.at(index);
  }
};

DependencyDataStore create_dependency_data_store(
    stream_ctx &ctx, const ExpandedDag &expanded_dag)
{
  DependencyDataStore dependency_data_store;
  dependency_data_store.expanded_dag = &expanded_dag;
  const std::size_t count =
      static_cast<std::size_t>(expanded_dag.task_graph.nb_fields) *
      static_cast<std::size_t>(expanded_dag.task_graph.max_width);
  dependency_data_store.slots.reserve(count);
  for (int field = 0; field < expanded_dag.task_graph.nb_fields; ++field) {
    for (long point = 0; point < expanded_dag.task_graph.max_width;
         ++point) {
      auto slot = ctx.logical_data<unsigned char>(
          expanded_dag.task_graph.output_bytes_per_task);
      slot.set_symbol(
          make_data_symbol(expanded_dag.dag_index(), field, point));
      dependency_data_store.slots.push_back(std::move(slot));
    }
  }
  return dependency_data_store;
}

std::string make_scratch_symbol(const TaskCoordinates &coordinates)
{
  return "S(" + std::to_string(coordinates.dag_index) + "," +
         std::to_string(coordinates.timestep) + "," +
         std::to_string(coordinates.point) + ")";
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const char *name)
{
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs + rhs;
}

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                               const char *name)
{
  if (lhs != 0 &&
      rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs * rhs;
}

std::uint64_t minimum_device_memory_bytes(
    const std::vector<ExpandedDag> &expanded_dags,
    const TaskPlacement &task_placement, int device_id)
{
  std::uint64_t result = 0;
  std::uint64_t max_scratch_bytes = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    std::uint64_t assigned_points = 0;
    for (long point = 0; point < expanded_dag.task_graph.max_width;
         ++point) {
      const TaskCoordinates coordinates{
          expanded_dag.dag_index(), 0, point};
      if (task_placement.device_for(
              expanded_dag.task_graph, coordinates) == device_id) {
        ++assigned_points;
      }
    }
    const std::uint64_t assigned_slots = checked_multiply(
        assigned_points,
        static_cast<std::uint64_t>(
            expanded_dag.task_graph.nb_fields),
        "minimum dependency-data slot count");
    const std::uint64_t dependency_data_bytes = checked_multiply(
        assigned_slots,
        static_cast<std::uint64_t>(
            expanded_dag.task_graph.output_bytes_per_task),
        "minimum dependency-data byte count");
    result = checked_add(
        result, dependency_data_bytes,
        "minimum device memory requirement");
    for (const DagTask &task : expanded_dag.tasks) {
      if (task_placement.device_for(
              expanded_dag.task_graph, task.coordinates) == device_id) {
        max_scratch_bytes = std::max(
            max_scratch_bytes,
            static_cast<std::uint64_t>(
                expanded_dag.task_graph.scratch_bytes_per_task));
        break;
      }
    }
  }
  return checked_add(
      result, max_scratch_bytes, "minimum device memory requirement");
}

void validate_device_memory(
    const std::vector<ExpandedDag> &expanded_dags,
    const TaskPlacement &task_placement)
{
  for (int device_id : task_placement.devices) {
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const std::uint64_t required_bytes =
        minimum_device_memory_bytes(
            expanded_dags, task_placement, device_id);
    if (required_bytes > free_bytes) {
      std::ostringstream message;
      message << "dependency data assigned to CUDA device " << device_id
              << " and one temporary scratch allocation require at least "
              << required_bytes << " bytes, but that device currently has "
              << free_bytes << " free bytes";
      throw std::runtime_error(message.str());
    }
  }
  cuda_check(
      cudaSetDevice(task_placement.devices.front()), "cudaSetDevice");
}

std::vector<GpuKernelResources> validate_gpu_kernel_configs_on_device(
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const CudaDeviceInfo &device)
{
  struct KernelSharedMemoryRequirement {
    KernelType workload;
    ComputeDataType compute_data_type;
    std::size_t bytes;
  };

  if (expanded_dags.size() != gpu_kernel_configs.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }

  std::vector<KernelSharedMemoryRequirement> shared_memory_requirements;
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const ExpandedDag &expanded_dag = expanded_dags[i];
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    const GpuKernelLaunchConfig &launch = gpu_kernel_config.launch;
    if (launch.blocks_per_task > device.max_grid_size_x) {
      throw std::runtime_error(
          "-cuda-blocks-per-task exceeds the selected device limit");
    }
    if (launch.threads_per_block > device.max_threads_per_block) {
      throw std::runtime_error(
          "-cuda-threads-per-block exceeds the selected device limit");
    }
    if (launch.dynamic_shared_memory_bytes >
        device.optin_shared_memory_per_block) {
      throw std::runtime_error(
          "-cuda-shmem-bytes-per-block exceeds the selected device limit");
    }

    const WorkloadKernelLimits limits =
        workload_kernel_limits(
            expanded_dag.task_graph.kernel.type,
            gpu_kernel_config.compute_data_type);
    if (launch.threads_per_block > limits.max_threads_per_block) {
      throw std::runtime_error(
          "-cuda-threads-per-block exceeds the selected kernel limit");
    }
    if (limits.static_shared_memory_bytes >
            device.optin_shared_memory_per_block ||
        launch.dynamic_shared_memory_bytes >
            device.optin_shared_memory_per_block -
                limits.static_shared_memory_bytes) {
      throw std::runtime_error(
          "-cuda-shmem-bytes-per-block exceeds the selected kernel limit");
    }

    const KernelType workload = expanded_dag.task_graph.kernel.type;
    const auto same_kernel =
        [workload, &gpu_kernel_config](
            const KernelSharedMemoryRequirement &requirement) {
          return requirement.workload == workload &&
                 (workload != KernelType::COMPUTE_BOUND ||
                  requirement.compute_data_type ==
                      gpu_kernel_config.compute_data_type);
        };
    auto requirement = std::find_if(
        shared_memory_requirements.begin(),
        shared_memory_requirements.end(), same_kernel);
    if (requirement == shared_memory_requirements.end()) {
      shared_memory_requirements.push_back(
          {workload, gpu_kernel_config.compute_data_type,
           launch.dynamic_shared_memory_bytes});
    } else {
      requirement->bytes =
          std::max(requirement->bytes,
                   launch.dynamic_shared_memory_bytes);
    }
  }

  for (const KernelSharedMemoryRequirement &requirement :
       shared_memory_requirements) {
    const WorkloadKernelLimits limits =
        workload_kernel_limits(requirement.workload,
                               requirement.compute_data_type);
    if (limits.static_shared_memory_bytes >
            device.legacy_shared_memory_per_block ||
        requirement.bytes >
            device.legacy_shared_memory_per_block -
                limits.static_shared_memory_bytes) {
      set_workload_dynamic_shared_memory_limit(
          requirement.workload, requirement.compute_data_type,
          requirement.bytes);
    }
  }

  std::vector<GpuKernelResources> result;
  result.reserve(expanded_dags.size());
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    GpuKernelResources resources = gpu_kernel_resources(
        expanded_dags[i].task_graph.kernel.type,
        gpu_kernel_configs[i].compute_data_type,
        gpu_kernel_configs[i].launch);
    resources.device_id = device.device_id;
    result.push_back(resources);
  }
  return result;
}

KernelResourcesByDag validate_gpu_kernel_configs(
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<CudaDeviceInfo> &devices)
{
  KernelResourcesByDag result(expanded_dags.size());
  for (const CudaDeviceInfo &device : devices) {
    cuda_check(cudaSetDevice(device.device_id), "cudaSetDevice");
    const std::vector<GpuKernelResources> device_resources =
        validate_gpu_kernel_configs_on_device(
            expanded_dags, gpu_kernel_configs, device);
    for (std::size_t dag_index = 0;
         dag_index < expanded_dags.size(); ++dag_index) {
      result[dag_index].push_back(device_resources[dag_index]);
    }
  }
  cuda_check(cudaSetDevice(devices.front().device_id), "cudaSetDevice");
  return result;
}

void submit_dag(stream_ctx &ctx, const ExpandedDag &expanded_dag,
                const TaskIterationCounts &task_iteration_counts,
                const GpuKernelConfig &gpu_kernel_config,
                DependencyDataStore &dependency_data_store,
                const TaskPlacement &task_placement)
{
  if (task_iteration_counts.size() != expanded_dag.tasks.size()) {
    throw std::logic_error(
        "task iteration count does not match DAG task count");
  }
  for (std::size_t task_index = 0;
       task_index < expanded_dag.tasks.size(); ++task_index) {
    const DagTask &dag_task = expanded_dag.tasks[task_index];
    const DataAccess output_access =
        output_data_access(expanded_dag, dag_task);
    if (output_access.mode != DataAccessMode::write) {
      throw std::logic_error("task output is not a write access");
    }
    const int device_id = task_placement.device_for(
        expanded_dag.task_graph, dag_task.coordinates);
    auto stf_task = ctx.task(exec_place::device(device_id));
    stf_task.add_deps(
        dependency_data_store.at(output_access.data).write());

    for (const Predecessor &predecessor :
         dag_task.predecessors) {
      const DataAccess input_access =
          input_data_access(expanded_dag, predecessor);
      if (input_access.mode != DataAccessMode::read) {
        throw std::logic_error("task input is not a read access");
      }
      stf_task.add_deps(
          dependency_data_store.at(input_access.data).read());
    }

    std::optional<LogicalByteData> scratch;
    if (uses_task_scratch(expanded_dag.task_graph)) {
      scratch.emplace(ctx.logical_data<unsigned char>(
          expanded_dag.task_graph.scratch_bytes_per_task));
      scratch->set_symbol(make_scratch_symbol(dag_task.coordinates));
      stf_task.add_deps(scratch->write());
    }
    stf_task.set_symbol(
        make_task_symbol(
            gpu_workload_name(expanded_dag.task_graph.kernel.type),
            dag_task.coordinates));
    attach_gpu_workload(
        stf_task, dag_task, expanded_dag.task_graph,
        task_iteration_counts[task_index],
        gpu_kernel_config);
  }
}

SampleResult run_sample(const std::vector<ExpandedDag> &expanded_dags,
                        const std::vector<TaskIterationCounts>
                            &task_iteration_counts,
                        const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                        const TaskPlacement &task_placement)
{
  if (task_iteration_counts.size() != expanded_dags.size()) {
    throw std::logic_error(
        "task iteration count set does not match DAG count");
  }
  if (task_placement.devices.empty()) {
    throw std::logic_error("sample task placement has no devices");
  }
  const int timing_device = task_placement.devices.front();
  const auto sample_start = Clock::now();
  cuda_check(cudaSetDevice(timing_device), "cudaSetDevice");

  const auto setup_start = Clock::now();
  CudaEvent start_event("cudaEventCreate start");
  CudaEvent stop_event("cudaEventCreate stop");
  stream_ctx ctx;

  std::vector<DependencyDataStore> dependency_data_stores;
  dependency_data_stores.reserve(expanded_dags.size());
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    dependency_data_stores.push_back(
        create_dependency_data_store(ctx, expanded_dag));
  }
  const auto setup_stop = Clock::now();

  cuda_check(cudaEventRecord(start_event.get(), ctx.fence()),
             "cudaEventRecord start");
  cuda_check(cudaEventSynchronize(start_event.get()),
             "cudaEventSynchronize start");
  const auto submission_start = Clock::now();
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    submit_dag(ctx, expanded_dags[i], task_iteration_counts[i],
               gpu_kernel_configs[i],
               dependency_data_stores[i], task_placement);
  }
  const auto submission_stop = Clock::now();
  cuda_check(cudaSetDevice(timing_device), "cudaSetDevice");
  cuda_check(cudaEventRecord(stop_event.get(), ctx.fence()),
             "cudaEventRecord stop");
  cuda_check(cudaEventSynchronize(stop_event.get()),
             "cudaEventSynchronize stop");
  ctx.finalize();
  cuda_check(cudaSetDevice(timing_device), "cudaSetDevice");
  cuda_check(cudaGetLastError(), "CUDASTF execution");

  float dag_makespan = 0.0f;
  cuda_check(cudaEventElapsedTime(
                 &dag_makespan, start_event.get(), stop_event.get()),
             "cudaEventElapsedTime");
  start_event.destroy_checked("cudaEventDestroy start");
  stop_event.destroy_checked("cudaEventDestroy stop");
  const auto sample_stop = Clock::now();

  SampleResult result;
  result.performance.submission_ms =
      milliseconds(submission_start, submission_stop);
  result.performance.dag_makespan_ms = dag_makespan;
  result.diagnostics.setup_ms =
      milliseconds(setup_start, setup_stop);
  result.diagnostics.sample_total_ms =
      milliseconds(sample_start, sample_stop);
  return result;
}

}  // namespace

int main(int argc, char **argv)
{
  try {
    Arguments arguments = parse_arguments(argc, argv);
    if (arguments.help_requested) {
      print_backend_help();
    }
    std::vector<char *> core_argv = arguments.core_argv();
    App task_bench_app(
        static_cast<int>(core_argv.size()), core_argv.data());
    if (arguments.gpu_kernel_configs.size() !=
        task_bench_app.graphs.size()) {
      throw std::logic_error(
          "GPU kernel configuration count does not match Task Bench graphs");
    }
    const std::vector<ExpandedDag> expanded_dags =
        expand_task_graphs(task_bench_app);
    const TopologyMetrics topology_metrics =
        compute_topology_metrics(expanded_dags);
    std::vector<TaskIterationCounts> task_iteration_counts;
    task_iteration_counts.reserve(expanded_dags.size());
    for (const ExpandedDag &expanded_dag : expanded_dags) {
      task_iteration_counts.push_back(
          make_task_iteration_counts(expanded_dag));
    }
    const std::vector<CudaDeviceInfo> devices =
        inspect_cuda_devices(arguments.run.task_placement);
    const KernelResourcesByDag kernel_resources =
        validate_gpu_kernel_configs(
            expanded_dags, arguments.gpu_kernel_configs, devices);
    validate_device_memory(
        expanded_dags, arguments.run.task_placement);
    const std::string execution_config_hash =
        compute_execution_config_hash(
            arguments.run, arguments.gpu_kernel_configs, expanded_dags);

    if (task_bench_app.verbose) {
      task_bench_app.display();
    }

    for (int i = 0; i < arguments.run.warmup_samples; ++i) {
      (void)run_sample(expanded_dags, task_iteration_counts,
                       arguments.gpu_kernel_configs,
                       arguments.run.task_placement);
    }
    std::vector<SampleResult> samples;
    samples.reserve(arguments.run.measured_samples);
    for (int i = 0; i < arguments.run.measured_samples; ++i) {
      samples.push_back(
          run_sample(expanded_dags, task_iteration_counts,
                     arguments.gpu_kernel_configs,
                     arguments.run.task_placement));
    }

    print_report(arguments.run, devices, expanded_dags,
                 task_iteration_counts, arguments.gpu_kernel_configs,
                 kernel_resources, topology_metrics,
                 execution_config_hash, samples);
    if (!arguments.output.run_json_path.empty()) {
      write_run_json(
          arguments.output.run_json_path, arguments.run, devices,
          arguments.core_arguments, expanded_dags,
          task_iteration_counts, arguments.gpu_kernel_configs,
          kernel_resources, execution_config_hash, samples);
    }
    if (!arguments.output.analysis_json_path.empty()) {
      write_analysis_json(
          arguments.output.analysis_json_path, expanded_dags,
          topology_metrics, execution_config_hash);
    }
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
