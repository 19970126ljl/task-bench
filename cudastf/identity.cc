#include "identity.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "results.h"

#ifndef TASKBENCH_CCCL_REVISION
#define TASKBENCH_CCCL_REVISION "unknown"
#endif

#ifndef TASKBENCH_REVISION
#define TASKBENCH_REVISION "unknown"
#endif

#ifndef TASKBENCH_CCCL_WORKTREE_DIRTY
#define TASKBENCH_CCCL_WORKTREE_DIRTY 0
#endif

#ifndef TASKBENCH_WORKTREE_DIRTY
#define TASKBENCH_WORKTREE_DIRTY 0
#endif

#ifndef TASKBENCH_BUILD_TYPE
#define TASKBENCH_BUILD_TYPE "unknown"
#endif

#ifndef TASKBENCH_CUDA_ARCH_RESOLVED
#define TASKBENCH_CUDA_ARCH_RESOLVED "unknown"
#endif

namespace {

class StableHash {
public:
  void add_u8(std::uint8_t value)
  {
    value_ ^= value;
    value_ *= UINT64_C(1099511628211);
  }

  template <typename T>
  void add_unsigned(T value)
  {
    static_assert(std::is_unsigned<T>::value, "unsigned integer required");
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      add_u8(static_cast<std::uint8_t>(value & 0xff));
      value >>= 8;
    }
  }

  void add_i64(std::int64_t value)
  {
    add_unsigned(static_cast<std::uint64_t>(value));
  }

  void add_u64(std::uint64_t value) { add_unsigned(value); }

  void add_f64(double value)
  {
    static_assert(sizeof(value) == sizeof(std::uint64_t),
                  "double must have a 64-bit representation");
    std::uint64_t bits = 0;
    if (value == 0.0) {
      value = 0.0;
    }
    std::memcpy(&bits, &value, sizeof(bits));
    add_u64(bits);
  }

  void add_string(const std::string &value)
  {
    add_u64(value.size());
    for (unsigned char byte : value) add_u8(byte);
  }

  std::string hex_digest() const
  {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value_;
    return out.str();
  }

private:
  std::uint64_t value_ = UINT64_C(14695981039346656037);
};

const char *kernel_type_name(KernelType type)
{
  switch (type) {
  case KernelType::EMPTY:
    return "empty";
  case KernelType::BUSY_WAIT:
    return "busy_wait";
  case KernelType::MEMORY_BOUND:
    return "memory_bound";
  case KernelType::COMPUTE_BOUND:
    return "compute_bound";
  default:
    throw std::logic_error("unsupported kernel type in execution identity");
  }
}

void add_effective_kernel_config(StableHash &hash,
                                 const TaskGraph &task_graph,
                                 const GpuKernelConfig &gpu_kernel_config)
{
  hash.add_string(kernel_type_name(task_graph.kernel.type));
  switch (task_graph.kernel.type) {
  case KernelType::EMPTY:
    break;
  case KernelType::BUSY_WAIT:
    hash.add_i64(task_graph.kernel.iterations);
    break;
  case KernelType::MEMORY_BOUND:
    hash.add_i64(task_graph.kernel.iterations);
    hash.add_i64(task_graph.kernel.samples);
    break;
  case KernelType::COMPUTE_BOUND:
    hash.add_i64(task_graph.kernel.iterations);
    hash.add_string(
        compute_data_type_name(gpu_kernel_config.compute_data_type));
    break;
  default:
    throw std::logic_error("unsupported kernel type in execution identity");
  }
  hash.add_f64(task_graph.kernel.imbalance);
}

}  // namespace

std::string compute_topology_hash(const std::vector<DagTask> &tasks)
{
  StableHash hash;
  hash.add_string("task-bench-dag-topology");
  hash.add_u64(tasks.size());
  for (const DagTask &task : tasks) {
    hash.add_i64(task.coordinates.timestep);
    hash.add_i64(task.coordinates.point);
    hash.add_u64(task.predecessors.size());
    for (const Predecessor &predecessor : task.predecessors) {
      hash.add_i64(predecessor.timestep);
      hash.add_i64(predecessor.point);
    }
  }
  return hash.hex_digest();
}

std::string compute_workload_config_hash(
    const RunConfig &run_config,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<ExpandedDag> &expanded_dags)
{
  if (gpu_kernel_configs.size() != expanded_dags.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }

  StableHash hash;
  hash.add_string("cudastf-workload-config");
  hash.add_string(run_config.context);
  hash.add_string(run_config.logical_data_allocator);
  if (run_config.stream_pool_size_per_device == 0) {
    throw std::logic_error("stream pool size was not resolved");
  }
  hash.add_u64(run_config.stream_pool_size_per_device);
  if (run_config.task_placement.devices.empty()) {
    throw std::logic_error("execution configuration has no CUDA devices");
  }
  hash.add_i64(run_config.task_placement.devices.front());
  if (run_config.task_placement.devices.size() > 1) {
    hash.add_string("multi-gpu-placement");
    hash.add_u64(run_config.task_placement.devices.size());
    for (int device : run_config.task_placement.devices) {
      hash.add_i64(device);
    }
    hash.add_string(task_placement_policy_name(
        run_config.task_placement.policy));
  }
  hash.add_u64(expanded_dags.size());

  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const ExpandedDag &expanded_dag = expanded_dags[i];
    const TaskGraph &task_graph = expanded_dag.task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    hash.add_string(expanded_dag.topology_hash);
    add_effective_kernel_config(
        hash, task_graph, gpu_kernel_config);
    hash.add_u64(task_graph.output_bytes_per_task);
    hash.add_u64(task_graph.scratch_bytes_per_task);
    hash.add_i64(task_graph.nb_fields);
    hash.add_i64(gpu_kernel_config.launch.blocks_per_task);
    hash.add_i64(gpu_kernel_config.launch.threads_per_block);
    hash.add_u64(
        gpu_kernel_config.launch.dynamic_shared_memory_bytes);
  }

  return hash.hex_digest();
}

std::string compute_execution_config_hash(
    const RunConfig &run_config,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<ExpandedDag> &expanded_dags)
{
  StableHash hash;
  hash.add_string("cudastf-execution-config");
  hash.add_string(compute_workload_config_hash(
      run_config, gpu_kernel_configs, expanded_dags));
  hash.add_string(cuda_feature_state_name(
      run_config.task_serialization));
  hash.add_string(cuda_feature_state_name(run_config.task_profiler));
  return hash.hex_digest();
}

std::string compute_environment_hash(
    const std::vector<CudaDeviceInfo> &devices)
{
  if (devices.empty()) {
    throw std::logic_error("execution environment has no CUDA devices");
  }
  StableHash hash;
  hash.add_string("cudastf-environment");
  hash.add_string(TASKBENCH_REVISION);
  hash.add_u8(TASKBENCH_WORKTREE_DIRTY ? 1 : 0);
  hash.add_string(TASKBENCH_CCCL_REVISION);
  hash.add_u8(TASKBENCH_CCCL_WORKTREE_DIRTY ? 1 : 0);
  hash.add_string(TASKBENCH_BUILD_TYPE);
  hash.add_string(TASKBENCH_CUDA_ARCH_RESOLVED);
  hash.add_i64(devices.front().runtime_version);
  hash.add_i64(devices.front().driver_version);
  hash.add_u64(devices.size());
  for (const CudaDeviceInfo &device : devices) {
    hash.add_string(device.name);
    hash.add_i64(device.compute_capability_major);
    hash.add_i64(device.compute_capability_minor);
    hash.add_i64(device.sm_count);
  }
  return hash.hex_digest();
}

std::string compute_raw_data_hash(
    const std::string &workload_config_hash,
    const std::string &execution_config_hash,
    const std::string &environment_hash,
    const KernelResourcesByDag &kernel_resources,
    const std::vector<SampleResult> &samples)
{
  StableHash hash;
  hash.add_string("cudastf-raw-data");
  hash.add_string(workload_config_hash);
  hash.add_string(execution_config_hash);
  hash.add_string(environment_hash);
  hash.add_u64(kernel_resources.size());
  for (const std::vector<GpuKernelResources> &dag_resources :
       kernel_resources) {
    hash.add_u64(dag_resources.size());
    for (const GpuKernelResources &resources : dag_resources) {
      hash.add_i64(resources.device_id);
      hash.add_i64(resources.registers_per_thread);
      hash.add_u64(resources.static_shared_memory_bytes);
      hash.add_i64(resources.max_active_blocks_per_sm);
    }
  }
  hash.add_u64(samples.size());
  for (const SampleResult &sample : samples) {
    hash.add_f64(sample.performance.submission_ms);
    hash.add_f64(sample.performance.dag_makespan_ms);
    hash.add_f64(sample.diagnostics.setup_ms);
    hash.add_f64(sample.diagnostics.sample_total_ms);
    hash.add_u8(sample.task_profile.has_value() ? 1 : 0);
    if (!sample.task_profile.has_value()) continue;

    const TaskProfileSample &profile = *sample.task_profile;
    hash.add_u64(profile.cupti_timestamp_origin_ns);
    hash.add_u64(profile.contexts.size());
    for (const TaskProfileContext &context : profile.contexts) {
      hash.add_u64(context.context_id);
      hash.add_string(context.label);
      hash.add_u8(context.has_gpu_activity ? 1 : 0);
      hash.add_u64(context.start_ns);
      hash.add_u64(context.end_ns);
      hash.add_f64(context.elapsed_ms);
      hash.add_u64(context.task_count);
      hash.add_u64(context.operation_count);
      hash.add_string(cuda_feature_state_name(context.task_serialization));
      hash.add_u64(context.regions.size());
      for (const TaskProfileRegion &region : context.regions) {
        hash.add_u64(region.region_id);
        hash.add_string(region.label);
        hash.add_u8(region.has_gpu_activity ? 1 : 0);
        hash.add_u64(region.start_ns);
        hash.add_u64(region.end_ns);
        hash.add_f64(region.elapsed_ms);
        hash.add_u64(region.task_count);
        hash.add_u64(region.operation_count);
      }
    }
    hash.add_u64(profile.tasks.size());
    for (const TaskProfileRecord &task : profile.tasks) {
      hash.add_i64(task.key.dag_index);
      hash.add_i64(task.key.timestep);
      hash.add_i64(task.key.point);
      hash.add_i64(task.configured_device);
      hash.add_u64(task.context_id);
      hash.add_u64(task.region_id);
      hash.add_i64(task.task_id);
      hash.add_string(task.symbol);
      hash.add_u8(task.has_gpu_activity ? 1 : 0);
      hash.add_u64(task.start_ns);
      hash.add_u64(task.end_ns);
      hash.add_f64(task.elapsed_ms);
      hash.add_u64(task.operation_count);
      hash.add_u64(task.device_timings.size());
      for (const TaskDeviceProfile &device : task.device_timings) {
        hash.add_i64(device.device_id);
        hash.add_u64(device.start_ns);
        hash.add_u64(device.end_ns);
        hash.add_f64(device.elapsed_ms);
        hash.add_u64(device.operation_count);
      }
    }
  }
  return hash.hex_digest();
}
