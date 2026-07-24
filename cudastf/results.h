#ifndef TASK_BENCH_CUDASTF_RESULTS_H
#define TASK_BENCH_CUDASTF_RESULTS_H

#include <string>
#include <vector>

#include "arguments.h"
#include "expanded_dag.h"
#include "workload.h"

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
};

struct PerformanceSummary {
  PerformanceSample minimum;
  PerformanceSample median;
  PerformanceSample p95;
};

PerformanceSummary summarize_performance(
    const std::vector<SampleResult> &samples);
void print_report(const RunConfig &run_config,
                  const CudaDeviceInfo &device,
                  const std::vector<ExpandedDag> &expanded_dags,
                  const std::vector<TaskIterationCounts>
                      &task_iteration_counts,
                  const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                  const std::vector<GpuKernelResources> &kernel_resources,
                  const std::string &execution_config_hash,
                  const std::vector<SampleResult> &samples);
void write_json(const std::string &path, const RunConfig &run_config,
                const CudaDeviceInfo &device,
                const std::vector<std::string> &core_arguments,
                const std::vector<ExpandedDag> &expanded_dags,
                const std::vector<TaskIterationCounts>
                    &task_iteration_counts,
                const std::vector<GpuKernelConfig> &gpu_kernel_configs,
                const std::vector<GpuKernelResources> &kernel_resources,
                const std::string &execution_config_hash,
                const std::vector<SampleResult> &samples);

#endif
