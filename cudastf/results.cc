#include "results.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "workload.h"
#include "derived_metrics.h"

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

#ifndef TASKBENCH_CUDA_ARCH
#define TASKBENCH_CUDA_ARCH "unknown"
#endif

#ifndef TASKBENCH_CUDA_ARCH_RESOLVED
#define TASKBENCH_CUDA_ARCH_RESOLVED "unknown"
#endif

namespace {

double percentile(std::vector<double> values, double fraction)
{
  std::sort(values.begin(), values.end());
  if (fraction <= 0.0) {
    return values.front();
  }
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

PerformanceSample performance_percentile(
    const std::vector<SampleResult> &samples, double fraction)
{
  PerformanceSample result;
  std::vector<double> values;
  values.reserve(samples.size());

#define TASK_BENCH_PERCENTILE(field)                                          \
  values.clear();                                                             \
  for (const SampleResult &sample : samples)                                  \
    values.push_back(sample.performance.field);                               \
  result.field = percentile(values, fraction)

  TASK_BENCH_PERCENTILE(submission_ms);
  TASK_BENCH_PERCENTILE(dag_makespan_ms);
#undef TASK_BENCH_PERCENTILE
  return result;
}

double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) {
    return values[middle];
  }
  return (values[middle - 1] + values[middle]) / 2.0;
}

PerformanceSample median_performance(
    const std::vector<SampleResult> &samples)
{
  PerformanceSample result;
  std::vector<double> values;
  values.reserve(samples.size());

#define TASK_BENCH_MEDIAN(field)                                              \
  values.clear();                                                             \
  for (const SampleResult &sample : samples)                                  \
    values.push_back(sample.performance.field);                               \
  result.field = median(values)

  TASK_BENCH_MEDIAN(submission_ms);
  TASK_BENCH_MEDIAN(dag_makespan_ms);
#undef TASK_BENCH_MEDIAN
  return result;
}

const char *dependence_name(DependenceType type)
{
  switch (type) {
  case DependenceType::TRIVIAL: return "trivial";
  case DependenceType::NO_COMM: return "no_comm";
  case DependenceType::STENCIL_1D: return "stencil_1d";
  case DependenceType::STENCIL_1D_PERIODIC: return "stencil_1d_periodic";
  case DependenceType::DOM: return "dom";
  case DependenceType::TREE: return "tree";
  case DependenceType::FFT: return "fft";
  case DependenceType::ALL_TO_ALL: return "all_to_all";
  case DependenceType::NEAREST: return "nearest";
  case DependenceType::SPREAD: return "spread";
  case DependenceType::RANDOM_NEAREST: return "random_nearest";
  case DependenceType::RANDOM_SPREAD: return "random_spread";
  }
  throw std::logic_error("unknown dependence type");
}

const char *kernel_name(KernelType type)
{
  switch (type) {
  case KernelType::EMPTY: return "empty";
  case KernelType::BUSY_WAIT: return "busy_wait";
  case KernelType::MEMORY_BOUND: return "memory_bound";
  case KernelType::COMPUTE_DGEMM: return "compute_dgemm";
  case KernelType::MEMORY_DAXPY: return "memory_daxpy";
  case KernelType::COMPUTE_BOUND: return "compute_bound";
  case KernelType::COMPUTE_BOUND2: return "compute_bound2";
  case KernelType::IO_BOUND: return "io_bound";
  case KernelType::LOAD_IMBALANCE: return "load_imbalance";
  }
  throw std::logic_error("unknown kernel type");
}

std::string json_escape(const std::string &input)
{
  std::string output;
  output.reserve(input.size() + 2);
  for (unsigned char c : input) {
    switch (c) {
    case '"': output += "\\\""; break;
    case '\\': output += "\\\\"; break;
    case '\b': output += "\\b"; break;
    case '\f': output += "\\f"; break;
    case '\n': output += "\\n"; break;
    case '\r': output += "\\r"; break;
    case '\t': output += "\\t"; break;
    default:
      if (c < 0x20) {
        const char hex[] = "0123456789abcdef";
        output += "\\u00";
        output += hex[(c >> 4) & 0xf];
        output += hex[c & 0xf];
      } else {
        output += static_cast<char>(c);
      }
    }
  }
  return output;
}

void write_sample(std::ostream &out, const SampleResult &sample)
{
  out << "{\"submission_ms\":" << sample.performance.submission_ms
      << ",\"dag_makespan_ms\":" << sample.performance.dag_makespan_ms
      << ",\"diagnostics\":{\"setup_ms\":"
      << sample.diagnostics.setup_ms
      << ",\"sample_total_ms\":" << sample.diagnostics.sample_total_ms
      << "},\"task_profile\":";
  if (!sample.task_profile.has_value()) {
    out << "null}";
    return;
  }

  const TaskProfileSample &profile = *sample.task_profile;
  out << "{\"cupti_timestamp_origin_ns\":"
      << profile.cupti_timestamp_origin_ns << ",\"contexts\":[";
  for (std::size_t i = 0; i < profile.contexts.size(); ++i) {
    if (i != 0) out << ",";
    const TaskProfileContext &context = profile.contexts[i];
    out << "{\"context_id\":" << context.context_id
        << ",\"label\":\"" << json_escape(context.label)
        << "\",\"has_gpu_activity\":"
        << (context.has_gpu_activity ? "true" : "false")
        << ",\"start_ns\":" << context.start_ns
        << ",\"end_ns\":" << context.end_ns
        << ",\"elapsed_ms\":" << context.elapsed_ms
        << ",\"task_count\":" << context.task_count
        << ",\"operation_count\":" << context.operation_count
        << ",\"task_serialization\":\""
        << cuda_feature_state_name(context.task_serialization)
        << "\",\"regions\":[";
    for (std::size_t j = 0; j < context.regions.size(); ++j) {
      if (j != 0) out << ",";
      const TaskProfileRegion &region = context.regions[j];
      out << "{\"region_id\":" << region.region_id
          << ",\"label\":\"" << json_escape(region.label)
          << "\",\"has_gpu_activity\":"
          << (region.has_gpu_activity ? "true" : "false")
          << ",\"start_ns\":" << region.start_ns
          << ",\"end_ns\":" << region.end_ns
          << ",\"elapsed_ms\":" << region.elapsed_ms
          << ",\"task_count\":" << region.task_count
          << ",\"operation_count\":" << region.operation_count << "}";
    }
    out << "]}";
  }
  out << "],\"tasks\":[";
  for (std::size_t i = 0; i < profile.tasks.size(); ++i) {
    if (i != 0) out << ",";
    const TaskProfileRecord &task = profile.tasks[i];
    out << "{\"dag_index\":" << task.key.dag_index
        << ",\"timestep\":" << task.key.timestep
        << ",\"point\":" << task.key.point
        << ",\"configured_device\":" << task.configured_device
        << ",\"context_id\":" << task.context_id
        << ",\"region_id\":" << task.region_id
        << ",\"task_id\":" << task.task_id
        << ",\"symbol\":\"" << json_escape(task.symbol)
        << "\",\"has_gpu_activity\":"
        << (task.has_gpu_activity ? "true" : "false")
        << ",\"start_ns\":" << task.start_ns
        << ",\"end_ns\":" << task.end_ns
        << ",\"elapsed_ms\":" << task.elapsed_ms
        << ",\"operation_count\":" << task.operation_count
        << ",\"device_timings\":[";
    for (std::size_t j = 0; j < task.device_timings.size(); ++j) {
      if (j != 0) out << ",";
      const TaskDeviceProfile &device = task.device_timings[j];
      out << "{\"device_id\":" << device.device_id
          << ",\"start_ns\":" << device.start_ns
          << ",\"end_ns\":" << device.end_ns
          << ",\"elapsed_ms\":" << device.elapsed_ms
          << ",\"operation_count\":" << device.operation_count << "}";
    }
    out << "]}";
  }
  out << "]}}";
}

void write_duration_statistics(
    std::ostream &out, const DurationStatistics &statistics)
{
  out << "{\"mean_ms\":" << statistics.mean_ms
      << ",\"median_ms\":" << statistics.median_ms
      << ",\"p95_ms\":" << statistics.p95_ms
      << ",\"cv\":" << statistics.cv << "}";
}

void write_parallelism_metrics(
    std::ostream &out, const ParallelismMetrics &metrics)
{
  out << "{\"work_ms\":" << metrics.work_ms
      << ",\"critical_path_ms\":" << metrics.critical_path_ms
      << ",\"average\":" << metrics.average
      << ",\"peak\":" << metrics.peak
      << ",\"p50\":" << metrics.p50
      << ",\"p95\":" << metrics.p95
      << ",\"cv\":" << metrics.cv << "}";
}

void write_parallelism_analysis(
    std::ostream &out, const ParallelismAnalysis &analysis)
{
  if (!analysis.available) {
    out << "{\"status\":\"unavailable\",\"reason\":\""
        << json_escape(analysis.unavailable_reason) << "\"}";
    return;
  }

  out << "{\"status\":\"available\",\"duration_aggregation\":\"median\""
      << ",\"duration_definition\":\"gpu_activity_envelope\""
      << ",\"dags\":[";
  for (std::size_t i = 0; i < analysis.dags.size(); ++i) {
    if (i != 0) out << ",";
    const DagParallelismMetrics &dag = analysis.dags[i];
    out << "{\"dag_index\":" << dag.dag_index << ",\"tasks\":[";
    for (std::size_t j = 0; j < dag.tasks.size(); ++j) {
      if (j != 0) out << ",";
      const MeasuredTaskDuration &task = dag.tasks[j];
      out << "{\"dag_index\":" << task.key.dag_index
          << ",\"timestep\":" << task.key.timestep
          << ",\"point\":" << task.key.point
          << ",\"configured_device\":" << task.configured_device
          << ",\"measured_duration_ms\":"
          << task.measured_duration_ms << "}";
    }
    out << "],\"task_duration\":";
    write_duration_statistics(out, dag.task_duration);
    out << ",\"parallelism\":";
    write_parallelism_metrics(out, dag.parallelism);
    out << "}";
  }
  out << "],\"combined\":{\"task_duration\":";
  write_duration_statistics(out, analysis.combined_task_duration);
  out << ",\"parallelism\":";
  write_parallelism_metrics(out, analysis.combined_parallelism);
  out << "}}";
}

void print_topology_metrics(const char *indent,
                            const DagTopologyMetrics &metrics)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(3)
      << indent << "Topology (unit task cost):\n"
      << indent << "  Critical path length: "
      << metrics.critical_path_length << "\n"
      << indent << "  Parallelism: average "
      << metrics.average_parallelism << ", peak "
      << metrics.peak_parallelism << ", p50 "
      << metrics.parallelism_p50 << ", p95 "
      << metrics.parallelism_p95 << ", CV "
      << metrics.parallelism_cv << "\n";
  std::cout << out.str();
}

void print_dag_summary(
    const char *indent, const DagTopologyMetrics &metrics,
    std::uint64_t dependency_data_capacity_bytes)
{
  std::cout << indent << "Tasks: " << metrics.tasks << "\n"
            << indent << "Dependency edges: "
            << metrics.dependency_edges << "\n"
            << indent << "Dependency-data capacity: "
            << dependency_data_capacity_bytes << " bytes\n";
  print_topology_metrics(indent, metrics);
}

void write_topology_metric_fields(
    std::ostream &out, const DagTopologyMetrics &metrics)
{
  out << "\"tasks\":" << metrics.tasks
      << ",\"dependency_edges\":" << metrics.dependency_edges
      << ",\"critical_path_length\":"
      << metrics.critical_path_length
      << ",\"parallelism\":{\"average\":"
      << metrics.average_parallelism << ",\"peak\":"
      << metrics.peak_parallelism << ",\"p50\":"
      << metrics.parallelism_p50 << ",\"p95\":"
      << metrics.parallelism_p95 << ",\"cv\":"
      << metrics.parallelism_cv << "}";
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
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs * rhs;
}

struct KernelWorkloadSummary {
  std::uint64_t total_iterations = 0;
  std::uint64_t min_iterations_per_task = 0;
  std::uint64_t max_iterations_per_task = 0;
  std::uint64_t scratch_window_bytes = 0;
  std::uint64_t scratch_read_bytes_per_iteration = 0;
  std::uint64_t scratch_write_bytes_per_iteration = 0;
  std::uint64_t total_scratch_read_bytes = 0;
  std::uint64_t total_scratch_write_bytes = 0;
};

KernelWorkloadSummary summarize_kernel_workload(
    const TaskGraph &task_graph,
    const TaskIterationCounts &task_iteration_counts)
{
  KernelWorkloadSummary result;
  bool have_task = false;
  for (std::uint64_t iterations : task_iteration_counts) {
    result.total_iterations =
        checked_add(result.total_iterations, iterations,
                    "workload iteration count");
    if (!have_task) {
      result.min_iterations_per_task = iterations;
      have_task = true;
    } else {
      result.min_iterations_per_task =
          std::min(result.min_iterations_per_task, iterations);
    }
    result.max_iterations_per_task =
        std::max(result.max_iterations_per_task, iterations);
  }

  if (uses_task_scratch(task_graph)) {
    const std::uint64_t samples =
        static_cast<std::uint64_t>(task_graph.kernel.samples);
    result.scratch_window_bytes =
        task_graph.scratch_bytes_per_task / samples;
    const std::uint64_t copy_bytes =
        result.scratch_window_bytes / 2;
    result.scratch_read_bytes_per_iteration = copy_bytes;
    result.scratch_write_bytes_per_iteration = copy_bytes;
    result.total_scratch_read_bytes =
        checked_multiply(result.total_iterations, copy_bytes,
                         "scratch read byte count");
    result.total_scratch_write_bytes =
        checked_multiply(result.total_iterations, copy_bytes,
                         "scratch write byte count");
  }
  return result;
}

std::uint64_t total_dependency_data_capacity_bytes(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::uint64_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = checked_add(
        result, expanded_dag.dependency_data_capacity_bytes,
        "dependency-data capacity");
  }
  return result;
}

void validate_dag_results(
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<TaskIterationCounts> &task_iteration_counts,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<CudaDeviceInfo> &devices,
    const KernelResourcesByDag &kernel_resources)
{
  if (devices.empty()) {
    throw std::logic_error("CUDA device result set is empty");
  }
  if (expanded_dags.size() != gpu_kernel_configs.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }
  if (expanded_dags.size() != task_iteration_counts.size()) {
    throw std::logic_error(
        "task iteration count set does not match DAG count");
  }
  if (expanded_dags.size() != kernel_resources.size()) {
    throw std::logic_error(
        "GPU kernel resource count does not match DAG count");
  }
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    if (expanded_dags[i].tasks.size() !=
        task_iteration_counts[i].size()) {
      throw std::logic_error(
          "task iteration count does not match DAG task count");
    }
    if (kernel_resources[i].size() != devices.size()) {
      throw std::logic_error(
          "GPU kernel resource device count does not match device count");
    }
    for (std::size_t device_index = 0;
         device_index < devices.size(); ++device_index) {
      if (kernel_resources[i][device_index].device_id !=
          devices[device_index].device_id) {
        throw std::logic_error(
            "GPU kernel resource device order does not match device order");
      }
    }
  }
}

void validate_topology_metrics(
    const std::vector<ExpandedDag> &expanded_dags,
    const TopologyMetrics &topology_metrics)
{
  if (topology_metrics.dags.size() != expanded_dags.size()) {
    throw std::logic_error(
        "topology metric count does not match DAG count");
  }
  std::uint64_t tasks = 0;
  std::uint64_t edges = 0;
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const DagTopologyMetrics &metrics = topology_metrics.dags[i];
    if (metrics.tasks != expanded_dags[i].tasks.size() ||
        metrics.dependency_edges !=
            expanded_dags[i].dependency_edges) {
      throw std::logic_error(
          "topology metrics do not match expanded DAG");
    }
    tasks = checked_add(tasks, metrics.tasks, "combined task count");
    edges = checked_add(
        edges, metrics.dependency_edges,
        "combined dependency edge count");
  }
  if (topology_metrics.combined.tasks != tasks ||
      topology_metrics.combined.dependency_edges != edges) {
    throw std::logic_error(
        "combined topology metrics do not match per-DAG metrics");
  }
}

}  // namespace

PerformanceSummary summarize_performance(
    const std::vector<SampleResult> &samples)
{
  if (samples.empty()) {
    throw std::runtime_error("cannot summarize an empty sample set");
  }
  return {performance_percentile(samples, 0.0),
          median_performance(samples),
          performance_percentile(samples, 0.95)};
}

void print_report(const RunConfig &run_config,
                  const std::vector<CudaDeviceInfo> &devices,
                  const std::vector<ExpandedDag> &expanded_dags,
                  const std::vector<TaskIterationCounts>
                      &task_iteration_counts,
                  const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                  const KernelResourcesByDag &kernel_resources,
                  const TopologyMetrics &topology_metrics,
                  const DerivedMetrics &derived_metrics,
                  const std::string &workload_config_hash,
                  const std::string &execution_config_hash,
                  const std::string &environment_hash,
                  const std::vector<SampleResult> &samples)
{
  validate_dag_results(
      expanded_dags, task_iteration_counts, gpu_kernel_configs,
      devices, kernel_resources);
  validate_topology_metrics(expanded_dags, topology_metrics);
  const PerformanceSummary summary = summarize_performance(samples);
  std::cout << "CUDASTF Task Bench\n"
            << "  Context: " << run_config.context << "\n"
            << "  Logical data allocator: "
            << run_config.logical_data_allocator << "\n"
            << "  Devices: ";
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << devices[i].device_id << " (" << devices[i].name << ")";
  }
  std::cout << "\n"
            << "  Task placement: ";
  if (devices.size() == 1) {
    std::cout << "all tasks on device " << devices.front().device_id;
  } else {
    std::cout
        << task_placement_policy_name(run_config.task_placement.policy)
        << " ("
        << task_placement_policy_description(
               run_config.task_placement.policy)
        << ")";
  }
  std::cout << "\n"
            << "  Task serialization: "
            << cuda_feature_state_name(run_config.task_serialization)
            << "\n"
            << "  Task profiler: "
            << cuda_feature_state_name(run_config.task_profiler)
            << "\n"
            << "  DAGs: " << expanded_dags.size() << "\n";
  if (expanded_dags.size() > 1) {
    std::cout << "  Combined DAGs:\n";
    print_dag_summary(
        "    ", topology_metrics.combined,
        total_dependency_data_capacity_bytes(expanded_dags));
  }
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const TaskGraph &task_graph = expanded_dags[i].task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    const KernelWorkloadSummary kernel_workload =
        summarize_kernel_workload(
            task_graph, task_iteration_counts[i]);
    std::cout << "  DAG " << i << ":\n";
    print_dag_summary(
        "    ", topology_metrics.dags[i],
        expanded_dags[i].dependency_data_capacity_bytes);
    std::cout << "    Kernel: "
              << kernel_name(task_graph.kernel.type);
    if (task_graph.kernel.type == KernelType::COMPUTE_BOUND) {
      std::cout << " ("
                << compute_data_type_name(
                    gpu_kernel_config.compute_data_type)
                << ")";
    }
    std::cout << "\n";
    if (task_graph.kernel.type != KernelType::EMPTY) {
      std::cout << "    Kernel workload iterations: ";
      if (task_graph.kernel.imbalance == 0.0) {
        std::cout << task_graph.kernel.iterations << "/task";
      } else {
        std::cout << "configured " << task_graph.kernel.iterations
                  << "/task, assigned "
                  << kernel_workload.min_iterations_per_task << ".."
                  << kernel_workload.max_iterations_per_task << "/task";
      }
      std::cout << ", total " << kernel_workload.total_iterations << "\n";
      if (task_graph.kernel.imbalance != 0.0) {
        std::cout << "    Iteration imbalance: "
                  << task_graph.kernel.imbalance << "\n";
      }
    }
    std::cout << "    Launch: "
              << gpu_kernel_config.launch.blocks_per_task
              << " blocks x "
              << gpu_kernel_config.launch.threads_per_block
              << " threads, "
              << gpu_kernel_config.launch.dynamic_shared_memory_bytes
              << " dynamic shared memory bytes\n";
    for (const GpuKernelResources &resources :
         kernel_resources[i]) {
      std::cout << "    Resources on device " << resources.device_id
                << ": registers/thread "
                << resources.registers_per_thread
                << ", static shared memory "
                << resources.static_shared_memory_bytes
                << " bytes, max active blocks/SM "
                << resources.max_active_blocks_per_sm
                << "\n";
    }
    if (uses_task_scratch(task_graph)) {
      std::cout << "    Scratch: "
                << task_graph.scratch_bytes_per_task
                << " allocated bytes/task, "
                << "sample window "
                << kernel_workload.scratch_window_bytes
                << " bytes, read "
                << kernel_workload.scratch_read_bytes_per_iteration
                << " and write "
                << kernel_workload.scratch_write_bytes_per_iteration
                << " bytes/iteration, total reads "
                << kernel_workload.total_scratch_read_bytes
                << " bytes, total writes "
                << kernel_workload.total_scratch_write_bytes << " bytes\n";
    }
  }
  std::cout << "  Workload config hash: "
            << workload_config_hash << "\n"
            << "  Execution config hash: "
            << execution_config_hash << "\n"
            << "  Environment hash: " << environment_hash << "\n"
            << std::fixed << std::setprecision(3)
            << "  Median submission: "
            << summary.median.submission_ms << " ms\n";
  if (run_config.task_serialization == CudaFeatureState::enabled) {
    std::cout << "  Median serialized collection makespan: "
              << summary.median.dag_makespan_ms << " ms\n"
              << "  Serialized collection makespan p95: "
              << summary.p95.dag_makespan_ms << " ms\n";
  } else {
    std::cout << "  Median DAG makespan: "
              << summary.median.dag_makespan_ms << " ms\n"
              << "  DAG makespan p95: " << summary.p95.dag_makespan_ms
              << " ms\n";
  }
  if (derived_metrics.parallelism.available) {
    const ParallelismMetrics &combined =
        derived_metrics.parallelism.combined_parallelism;
    std::cout << "  Measured task work: " << combined.work_ms << " ms\n"
              << "  Weighted critical path: "
              << combined.critical_path_ms << " ms\n"
              << "  DAG parallelism: average "
              << combined.average << ", peak " << combined.peak
              << ", p50 " << combined.p50 << ", p95 "
              << combined.p95 << ", CV " << combined.cv << "\n";
  }
}

void write_run_json(
    const std::string &path, const RunConfig &run_config,
    const std::vector<CudaDeviceInfo> &devices,
    const std::vector<std::string> &core_arguments,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<TaskIterationCounts> &task_iteration_counts,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const KernelResourcesByDag &kernel_resources,
    const std::string &workload_config_hash,
    const std::string &execution_config_hash,
    const std::string &environment_hash,
    const std::string &raw_data_hash,
    const std::vector<SampleResult> &samples)
{
  validate_dag_results(
      expanded_dags, task_iteration_counts, gpu_kernel_configs,
      devices, kernel_resources);
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open run JSON output: " + path);
  }
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\n"
      << "  \"format\":\"cudastf-task-bench-run\",\n"
      << "  \"schema_version\":3,\n"
      << "  \"backend\":\"cudastf\",\n"
      << "  \"task_bench_revision\":\""
      << json_escape(TASKBENCH_REVISION) << "\",\n"
      << "  \"task_bench_worktree_dirty\":"
      << (TASKBENCH_WORKTREE_DIRTY ? "true" : "false") << ",\n"
      << "  \"cccl_revision\":\"" << json_escape(TASKBENCH_CCCL_REVISION)
      << "\",\n"
      << "  \"cccl_worktree_dirty\":"
      << (TASKBENCH_CCCL_WORKTREE_DIRTY ? "true" : "false") << ",\n"
      << "  \"build\":{\"type\":\"" << json_escape(TASKBENCH_BUILD_TYPE)
      << "\",\"cuda_arch_option\":\"" << json_escape(TASKBENCH_CUDA_ARCH)
      << "\",\"cuda_arch_resolved\":\""
      << json_escape(TASKBENCH_CUDA_ARCH_RESOLVED)
      << "\"},\n"
      << "  \"cuda_runtime_version\":" << devices.front().runtime_version
      << ",\n"
      << "  \"cuda_driver_version\":" << devices.front().driver_version
      << ",\n"
      << "  \"devices\":[";
  for (std::size_t i = 0; i < devices.size(); ++i) {
    if (i != 0) out << ",";
    const CudaDeviceInfo &device = devices[i];
    out << "{\"device_id\":" << device.device_id << ",\"name\":\""
        << json_escape(device.name) << "\",\"uuid\":\""
        << json_escape(device.uuid) << "\",\"compute_capability\":\""
        << device.compute_capability_major << "."
        << device.compute_capability_minor
        << "\",\"legacy_shared_memory_per_block_bytes\":"
        << device.legacy_shared_memory_per_block
        << ",\"optin_shared_memory_per_block_bytes\":"
        << device.optin_shared_memory_per_block << "}";
  }
  out << "],\n";

  out << "  \"core_arguments\":[";
  for (std::size_t i = 0; i < core_arguments.size(); ++i) {
    if (i != 0) out << ",";
    out << "\"" << json_escape(core_arguments[i]) << "\"";
  }
  out << "],\n"
      << "  \"run_config\":{\"device_ids\":[";
  for (std::size_t i = 0;
       i < run_config.task_placement.devices.size(); ++i) {
    if (i != 0) out << ",";
    out << run_config.task_placement.devices[i];
  }
  out << "],\"task_placement\":\""
      << task_placement_policy_name(run_config.task_placement.policy)
      << "\",\"context\":\"" << json_escape(run_config.context)
      << "\",\"logical_data_allocator\":\""
      << json_escape(run_config.logical_data_allocator)
      << "\",\"warmup_samples\":" << run_config.warmup_samples
      << ",\"measured_samples\":" << run_config.measured_samples
      << ",\"task_serialization\":\""
      << cuda_feature_state_name(run_config.task_serialization)
      << "\",\"task_profiler\":\""
      << cuda_feature_state_name(run_config.task_profiler)
      << "\"},\n"
      << "  \"workload_config_hash\":\""
      << workload_config_hash << "\",\n"
      << "  \"execution_config_hash\":\""
      << execution_config_hash << "\",\n"
      << "  \"environment_hash\":\"" << environment_hash << "\",\n"
      << "  \"raw_data_hash\":\"" << raw_data_hash << "\",\n";

  out << "  \"dags\":[\n";
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const ExpandedDag &expanded_dag = expanded_dags[i];
    const TaskGraph &task_graph = expanded_dag.task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    out << "    {\"dag_index\":" << expanded_dag.dag_index()
        << ",\"timesteps\":" << task_graph.timesteps
        << ",\"max_width\":" << task_graph.max_width
        << ",\"dependence\":\"" << dependence_name(task_graph.dependence)
        << "\",\"radix\":" << task_graph.radix
        << ",\"period\":" << task_graph.period
        << ",\"fraction_connected\":" << task_graph.fraction_connected
        << ",\"nb_fields\":" << task_graph.nb_fields
        << ",\"output_bytes_per_task\":"
        << task_graph.output_bytes_per_task
        << ",\"scratch_bytes_per_task\":"
        << task_graph.scratch_bytes_per_task
        << ",\"kernel\":{\"type\":\"" << kernel_name(task_graph.kernel.type)
        << "\",\"iterations\":" << task_graph.kernel.iterations
        << ",\"samples\":" << task_graph.kernel.samples
        << ",\"imbalance\":" << task_graph.kernel.imbalance
        << ",\"compute_data_type\":";
    if (task_graph.kernel.type == KernelType::COMPUTE_BOUND) {
      out << "\"" << compute_data_type_name(
          gpu_kernel_config.compute_data_type) << "\"";
    } else {
      out << "null";
    }
    out << ",\"launch\":{\"blocks_per_task\":"
        << gpu_kernel_config.launch.blocks_per_task
        << ",\"threads_per_block\":"
        << gpu_kernel_config.launch.threads_per_block
        << ",\"dynamic_shared_memory_bytes\":"
        << gpu_kernel_config.launch.dynamic_shared_memory_bytes
        << "},\"resources\":[";
    for (std::size_t resource_index = 0;
         resource_index < kernel_resources[i].size();
         ++resource_index) {
      if (resource_index != 0) out << ",";
      const GpuKernelResources &resources =
          kernel_resources[i][resource_index];
      out << "{\"device_id\":" << resources.device_id
          << ",\"registers_per_thread\":"
          << resources.registers_per_thread
          << ",\"static_shared_memory_bytes\":"
          << resources.static_shared_memory_bytes
          << ",\"max_active_blocks_per_sm\":"
          << resources.max_active_blocks_per_sm << "}";
    }
    out << "]}"
        << ",\"topology_hash\":\"" << expanded_dag.topology_hash
        << "\",\"tasks\":" << expanded_dag.tasks.size()
        << ",\"dependency_edges\":"
        << expanded_dag.dependency_edges << ",\"task_table\":[";
    for (std::size_t task_index = 0;
         task_index < expanded_dag.tasks.size(); ++task_index) {
      if (task_index != 0) out << ",";
      const DagTask &task = expanded_dag.tasks[task_index];
      out << "{\"dag_index\":" << task.coordinates.dag_index
          << ",\"timestep\":" << task.coordinates.timestep
          << ",\"point\":" << task.coordinates.point
          << ",\"configured_device\":"
          << run_config.task_placement.device_for(
                 expanded_dag.task_graph, task.coordinates)
          << ",\"iterations\":"
          << task_iteration_counts[i][task_index]
          << ",\"predecessors\":[";
      for (std::size_t predecessor_index = 0;
           predecessor_index < task.predecessors.size();
           ++predecessor_index) {
        if (predecessor_index != 0) out << ",";
        const Predecessor &predecessor =
            task.predecessors[predecessor_index];
        out << "{\"timestep\":" << predecessor.timestep
            << ",\"point\":" << predecessor.point << "}";
      }
      out << "]}";
    }
    out << "]}";
    out << (i + 1 == expanded_dags.size() ? "\n" : ",\n");
  }
  out << "  ],\n"
      << "  \"samples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) out << ",";
    write_sample(out, samples[i]);
  }
  out << "]\n}\n";

  if (!out) {
    throw std::runtime_error(
        "failed while writing run JSON output: " + path);
  }
}

void write_analysis_json(
    const std::string &path,
    const std::vector<ExpandedDag> &expanded_dags,
    const TopologyMetrics &topology_metrics,
    const DerivedMetrics &derived_metrics,
    const std::string &workload_config_hash,
    const std::string &execution_config_hash,
    const std::string &environment_hash,
    const std::string &raw_data_hash)
{
  validate_topology_metrics(expanded_dags, topology_metrics);
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error(
        "failed to open analysis JSON output: " + path);
  }

  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\n"
      << "  \"format\":\"cudastf-task-bench-analysis\",\n"
      << "  \"schema_version\":3,\n"
      << "  \"backend\":\"cudastf\",\n"
      << "  \"source\":{\"task_bench_revision\":\""
      << json_escape(TASKBENCH_REVISION)
      << "\",\"task_bench_worktree_dirty\":"
      << (TASKBENCH_WORKTREE_DIRTY ? "true" : "false")
      << ",\"workload_config_hash\":\""
      << workload_config_hash << "\""
      << ",\"execution_config_hash\":\""
      << execution_config_hash << "\",\"raw_data_hash\":\""
      << raw_data_hash << "\",\"environment_hash\":\""
      << environment_hash << "\",\"dags\":[";
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    if (i != 0) out << ",";
    out << "{\"dag_index\":" << expanded_dags[i].dag_index()
        << ",\"topology_hash\":\""
        << expanded_dags[i].topology_hash << "\"}";
  }
  out << "]},\n"
      << "  \"topology\":{\"dags\":[";
  for (std::size_t i = 0; i < topology_metrics.dags.size(); ++i) {
    if (i != 0) out << ",";
    out << "{\"dag_index\":" << expanded_dags[i].dag_index() << ",";
    write_topology_metric_fields(out, topology_metrics.dags[i]);
    out << "}";
  }
  out << "],\"combined\":{";
  write_topology_metric_fields(out, topology_metrics.combined);
  out << "}},\n  \"derived_metrics\":{\"parallelism\":";
  write_parallelism_analysis(out, derived_metrics.parallelism);
  out << "}\n}\n";

  if (!out) {
    throw std::runtime_error(
        "failed while writing analysis JSON output: " + path);
  }
}
