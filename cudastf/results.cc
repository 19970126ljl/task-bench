#include "results.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "workload.h"

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

void write_performance_metrics(std::ostream &out,
                               const PerformanceSample &sample)
{
  out << "{\"submission_ms\":" << sample.submission_ms
      << ",\"dag_makespan_ms\":" << sample.dag_makespan_ms << "}";
}

void write_sample(std::ostream &out, const SampleResult &sample)
{
  out << "{\"submission_ms\":" << sample.performance.submission_ms
      << ",\"dag_makespan_ms\":" << sample.performance.dag_makespan_ms
      << ",\"diagnostics\":{\"setup_ms\":"
      << sample.diagnostics.setup_ms
      << ",\"sample_total_ms\":" << sample.diagnostics.sample_total_ms
      << "}}";
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const char *name)
{
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs + rhs;
}

std::uint64_t total_tasks(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::uint64_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = checked_add(result, expanded_dag.tasks.size(),
                         "task count");
  }
  return result;
}

std::uint64_t total_edges(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::uint64_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = checked_add(result, expanded_dag.dependency_edges,
                         "dependency edge count");
  }
  return result;
}

std::uint64_t total_task_data_accesses(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::uint64_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = checked_add(result, expanded_dag.task_data_accesses,
                         "task data access count");
  }
  return result;
}

std::uint64_t total_task_data_store_bytes(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::uint64_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = checked_add(result, expanded_dag.task_data_store_bytes,
                         "task data store byte count");
  }
  return result;
}

std::size_t max_fanin(
    const std::vector<ExpandedDag> &expanded_dags)
{
  std::size_t result = 0;
  for (const ExpandedDag &expanded_dag : expanded_dags) {
    result = std::max(result, expanded_dag.max_fanin);
  }
  return result;
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
                  const CudaDeviceInfo &device,
                  const std::vector<ExpandedDag> &expanded_dags,
                  const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                  const std::string &execution_config_hash,
                  const std::vector<SampleResult> &samples)
{
  if (expanded_dags.size() != gpu_kernel_configs.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }
  const PerformanceSummary summary = summarize_performance(samples);
  std::cout << "CUDASTF Task Bench\n"
            << "  Context: " << run_config.context << "\n"
            << "  Logical data allocator: "
            << run_config.logical_data_allocator << "\n"
            << "  Device: " << device.device_id << " (" << device.name
            << ")\n"
            << "  DAGs: " << expanded_dags.size() << "\n"
            << "  Tasks: " << total_tasks(expanded_dags) << "\n"
            << "  Dependency edges: " << total_edges(expanded_dags) << "\n"
            << "  Task data accesses: "
            << total_task_data_accesses(expanded_dags) << "\n"
            << "  Max fan-in: " << max_fanin(expanded_dags) << "\n"
            << "  Task data store bytes: "
            << total_task_data_store_bytes(expanded_dags) << "\n";
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const TaskGraph &task_graph = expanded_dags[i].task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    const WorkloadModel model = workload_model(expanded_dags[i]);
    std::cout << "  DAG " << i << " kernel: "
              << kernel_name(task_graph.kernel.type)
              << ", iterations=" << task_graph.kernel.iterations
              << ", launch="
              << gpu_kernel_config.launch.blocks_per_task << "x"
              << gpu_kernel_config.launch.threads_per_block
              << ", dynamic shared memory="
              << gpu_kernel_config.launch.dynamic_shared_memory_bytes
              << " bytes"
              << ", logical iterations="
              << model.logical_iterations
              << ", task input reads="
              << model.task_input_read_bytes
              << " bytes"
              << ", task output writes="
              << model.task_output_write_bytes << " bytes"
              << ", scratch reads=" << model.scratch_read_bytes
              << " bytes"
              << ", scratch writes=" << model.scratch_write_bytes
              << " bytes";
    if (task_graph.kernel.type == KernelType::COMPUTE_BOUND) {
      std::cout << ", compute data type="
                << compute_data_type_name(
                    gpu_kernel_config.compute_data_type);
    }
    std::cout << "\n";
  }
  std::cout << "  Execution config hash: "
            << execution_config_hash << "\n"
            << std::fixed << std::setprecision(3)
            << "  Median submission: "
            << summary.median.submission_ms << " ms\n"
            << "  Median DAG makespan: "
            << summary.median.dag_makespan_ms << " ms\n"
            << "  DAG makespan p95: " << summary.p95.dag_makespan_ms
            << " ms\n";
}

void write_json(const std::string &path, const RunConfig &run_config,
                const CudaDeviceInfo &device,
                const std::vector<std::string> &core_arguments,
                const std::vector<ExpandedDag> &expanded_dags,
                const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                const std::string &execution_config_hash,
                const std::vector<SampleResult> &samples)
{
  if (expanded_dags.size() != gpu_kernel_configs.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to open JSON output: " + path);
  }
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\n"
      << "  \"format\":\"cudastf-task-bench-results\",\n"
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
      << "  \"cuda_runtime_version\":" << device.runtime_version << ",\n"
      << "  \"cuda_driver_version\":" << device.driver_version << ",\n"
      << "  \"device\":{\"device_id\":" << device.device_id
      << ",\"name\":\""
      << json_escape(device.name) << "\",\"uuid\":\""
      << json_escape(device.uuid) << "\",\"compute_capability\":\""
      << device.compute_capability_major << "."
      << device.compute_capability_minor
      << "\",\"legacy_shared_memory_per_block_bytes\":"
      << device.legacy_shared_memory_per_block
      << ",\"optin_shared_memory_per_block_bytes\":"
      << device.optin_shared_memory_per_block << "},\n";

  out << "  \"core_arguments\":[";
  for (std::size_t i = 0; i < core_arguments.size(); ++i) {
    if (i != 0) out << ",";
    out << "\"" << json_escape(core_arguments[i]) << "\"";
  }
  out << "],\n"
      << "  \"run_config\":{\"device_id\":" << run_config.device_id
      << ",\"context\":\"" << json_escape(run_config.context)
      << "\",\"logical_data_allocator\":\""
      << json_escape(run_config.logical_data_allocator)
      << "\",\"warmup_samples\":" << run_config.warmup_samples
      << ",\"measured_samples\":" << run_config.measured_samples
      << "},\n"
      << "  \"execution_config_hash\":\""
      << execution_config_hash << "\",\n";

  out << "  \"dags\":[\n";
  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const ExpandedDag &expanded_dag = expanded_dags[i];
    const TaskGraph &task_graph = expanded_dag.task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    const WorkloadModel model = workload_model(expanded_dag);
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
        << "},\"work_model\":{\"logical_iterations\":"
        << model.logical_iterations
        << ",\"task_input_read_bytes\":"
        << model.task_input_read_bytes
        << ",\"task_output_write_bytes\":"
        << model.task_output_write_bytes
        << ",\"scratch_read_bytes\":" << model.scratch_read_bytes
        << ",\"scratch_write_bytes\":" << model.scratch_write_bytes
        << "}}"
        << ",\"topology_hash\":\"" << expanded_dag.topology_hash
        << "\",\"tasks\":" << expanded_dag.tasks.size()
        << ",\"dependency_edges\":"
        << expanded_dag.dependency_edges
        << ",\"task_data_accesses\":"
        << expanded_dag.task_data_accesses
        << ",\"max_fanin\":"
        << expanded_dag.max_fanin
        << ",\"task_data_store_bytes\":"
        << expanded_dag.task_data_store_bytes
        << "}";
    out << (i + 1 == expanded_dags.size() ? "\n" : ",\n");
  }
  out << "  ],\n"
      << "  \"aggregate_counts\":{\"tasks\":"
      << total_tasks(expanded_dags) << ",\"dependency_edges\":"
      << total_edges(expanded_dags) << ",\"task_data_accesses\":"
      << total_task_data_accesses(expanded_dags)
      << ",\"task_data_store_bytes\":"
      << total_task_data_store_bytes(expanded_dags) << "},\n"
      << "  \"samples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i != 0) out << ",";
    write_sample(out, samples[i]);
  }
  const PerformanceSummary summary = summarize_performance(samples);
  out << "],\n  \"summary\":{\"min\":";
  write_performance_metrics(out, summary.minimum);
  out << ",\"median\":";
  write_performance_metrics(out, summary.median);
  out << ",\"p95\":";
  write_performance_metrics(out, summary.p95);
  out << "}\n}\n";

  if (!out) {
    throw std::runtime_error("failed while writing JSON output: " + path);
  }
}
