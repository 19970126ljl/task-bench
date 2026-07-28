#ifndef TASK_BENCH_CUDASTF_RESULTS_H
#define TASK_BENCH_CUDASTF_RESULTS_H

#include <optional>
#include <string>
#include <vector>

#include "arguments.h"
#include "expanded_dag.h"
#include "task_profile.h"
#include "topology.h"
#include "workload.h"

struct DerivedMetrics;

struct CudaDeviceInfo {
  int device_id = 0;
  std::string name;
  std::string uuid;
  int runtime_version = 0;
  int driver_version = 0;
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  int max_threads_per_block = 0;
  int max_grid_size_x = 0;
  std::size_t legacy_shared_memory_per_block = 0;
  std::size_t optin_shared_memory_per_block = 0;
};

struct PerformanceSample {
  double submission_ms = 0.0;
  double dag_makespan_ms = 0.0;
};

struct SampleDiagnostics {
  double setup_ms = 0.0;
  double sample_total_ms = 0.0;
};

struct SampleResult {
  PerformanceSample performance;
  SampleDiagnostics diagnostics;
  std::optional<TaskProfileSample> task_profile;
};

struct PerformanceSummary {
  PerformanceSample minimum;
  PerformanceSample median;
  PerformanceSample p95;
};

PerformanceSummary summarize_performance(
    const std::vector<SampleResult> &samples);
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
                  const std::vector<SampleResult> &samples);
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
    const std::vector<SampleResult> &samples);
void write_analysis_json(
    const std::string &path,
    const std::vector<ExpandedDag> &expanded_dags,
    const TopologyMetrics &topology_metrics,
    const DerivedMetrics &derived_metrics,
    const std::string &workload_config_hash,
    const std::string &execution_config_hash,
    const std::string &environment_hash,
    const std::string &raw_data_hash);

#endif
