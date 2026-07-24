#ifndef TASK_BENCH_CUDASTF_ARGUMENTS_H
#define TASK_BENCH_CUDASTF_ARGUMENTS_H

#include <cstddef>
#include <string>
#include <vector>

struct GpuKernelLaunchConfig {
  int blocks_per_task = 32;
  int threads_per_block = 128;
  std::size_t dynamic_shared_memory_bytes = 0;
};

enum class ComputeDataType {
  fp32,
  fp64,
};

struct GpuKernelConfig {
  GpuKernelLaunchConfig launch;
  ComputeDataType compute_data_type = ComputeDataType::fp32;
};

struct RunConfig {
  int device_id = 0;
  int warmup_samples = 1;
  int measured_samples = 5;
  std::string context = "stream";
  std::string logical_data_allocator = "cached";
  std::string json_path;
};

struct Arguments {
  RunConfig run;
  std::vector<GpuKernelConfig> gpu_kernel_configs;
  std::vector<std::string> core_arguments;
  bool help_requested = false;

  std::vector<char *> core_argv();
};

Arguments parse_arguments(int argc, char **argv);
const char *compute_data_type_name(ComputeDataType type);
void print_backend_help();

#endif
