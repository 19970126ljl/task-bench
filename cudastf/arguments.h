#ifndef TASK_BENCH_CUDASTF_ARGUMENTS_H
#define TASK_BENCH_CUDASTF_ARGUMENTS_H

#include <cstddef>
#include <string>
#include <vector>

#include "task_placement.h"

struct GpuKernelLaunchConfig {
  int blocks_per_task = 32;
  int threads_per_block = 128;
  std::size_t dynamic_shared_memory_bytes = 0;
};

enum class ComputeDataType {
  fp32,
  fp64,
};

enum class CudaFeatureState {
  disabled,
  enabled,
};

struct GpuKernelConfig {
  GpuKernelLaunchConfig launch;
  ComputeDataType compute_data_type = ComputeDataType::fp32;
};

struct RunConfig {
  TaskPlacement task_placement;
  // Resolved from the maximum DAG width after core argument parsing.
  std::size_t stream_pool_size_per_device = 0;
  int warmup_samples = 1;
  int measured_samples = 5;
  std::string context = "stream";
  std::string logical_data_allocator = "cached";
  CudaFeatureState task_serialization = CudaFeatureState::disabled;
  CudaFeatureState task_profiler = CudaFeatureState::disabled;
};

struct OutputConfig {
  std::string run_json_path;
  std::string analysis_json_path;
};

struct Arguments {
  RunConfig run;
  OutputConfig output;
  std::vector<GpuKernelConfig> gpu_kernel_configs;
  std::vector<std::string> core_arguments;
  bool help_requested = false;

  std::vector<char *> core_argv();
};

Arguments parse_arguments(int argc, char **argv);
std::size_t resolve_stream_pool_size_per_device(
    std::size_t configured_size, const std::vector<long> &dag_widths);
const char *compute_data_type_name(ComputeDataType type);
const char *cuda_feature_state_name(CudaFeatureState state);
void print_backend_help();

#endif
