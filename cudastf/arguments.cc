#include "arguments.h"

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

int parse_nonnegative_int(const char *flag, const char *text)
{
  errno = 0;
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 0 ||
      value > INT_MAX) {
    throw std::runtime_error(std::string("invalid value for ") + flag +
                             ": " + text);
  }
  return static_cast<int>(value);
}

int parse_positive_int(const char *flag, const char *text)
{
  const int value = parse_nonnegative_int(flag, text);
  if (value == 0) {
    throw std::runtime_error(std::string(flag) +
                             " must be greater than zero");
  }
  return value;
}

std::size_t parse_size(const char *flag, const char *text)
{
  if (text[0] == '-') {
    throw std::runtime_error(std::string("invalid value for ") + flag +
                             ": " + text);
  }
  errno = 0;
  char *end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' ||
      value > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error(std::string("invalid value for ") + flag +
                             ": " + text);
  }
  return static_cast<std::size_t>(value);
}

const char *require_value(int &i, int argc, char **argv)
{
  if (i + 1 >= argc) {
    throw std::runtime_error(std::string("flag requires an argument: ") +
                             argv[i]);
  }
  return argv[++i];
}

ComputeDataType parse_compute_data_type(const char *flag, const char *text)
{
  if (!std::strcmp(text, "fp32")) return ComputeDataType::fp32;
  if (!std::strcmp(text, "fp64")) return ComputeDataType::fp64;
  throw std::runtime_error(std::string("invalid value for ") + flag +
                           ": " + text + "; expected fp32 or fp64");
}

std::string logical_data_allocator_from_environment()
{
  const char *value = std::getenv("CUDASTF_DEFAULT_ALLOCATOR");
  if (value == nullptr) return "cached";
  if (!std::strcmp(value, "cached") ||
      !std::strcmp(value, "cached_fifo") ||
      !std::strcmp(value, "uncached") ||
      !std::strcmp(value, "pooled")) {
    return value;
  }
  throw std::runtime_error(
      std::string("invalid CUDASTF_DEFAULT_ALLOCATOR: ") + value);
}

}  // namespace

std::vector<char *> Arguments::core_argv()
{
  std::vector<char *> result;
  result.reserve(core_arguments.size());
  for (std::string &argument : core_arguments) {
    result.push_back(argument.data());
  }
  return result;
}

Arguments parse_arguments(int argc, char **argv)
{
  Arguments arguments;
  arguments.run.logical_data_allocator =
      logical_data_allocator_from_environment();
  GpuKernelConfig gpu_kernel_config;
  arguments.core_arguments.reserve(argc);
  arguments.core_arguments.emplace_back(
      argc > 0 ? argv[0] : "task_bench");

  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];
    if (!std::strcmp(arg, "-cuda-device")) {
      arguments.run.device_id =
          parse_nonnegative_int(arg, require_value(i, argc, argv));
    } else if (!std::strcmp(arg, "-cuda-warmup")) {
      arguments.run.warmup_samples =
          parse_nonnegative_int(arg, require_value(i, argc, argv));
    } else if (!std::strcmp(arg, "-cuda-runs")) {
      arguments.run.measured_samples =
          parse_nonnegative_int(arg, require_value(i, argc, argv));
      if (arguments.run.measured_samples == 0) {
        throw std::runtime_error("-cuda-runs must be greater than zero");
      }
    } else if (!std::strcmp(arg, "-cuda-context")) {
      arguments.run.context = require_value(i, argc, argv);
    } else if (!std::strcmp(arg, "-cuda-json")) {
      arguments.run.json_path = require_value(i, argc, argv);
      if (arguments.run.json_path.empty()) {
        throw std::runtime_error("-cuda-json requires a non-empty path");
      }
    } else if (!std::strcmp(arg, "-cuda-blocks-per-task")) {
      gpu_kernel_config.launch.blocks_per_task =
          parse_positive_int(arg, require_value(i, argc, argv));
    } else if (!std::strcmp(arg, "-cuda-threads-per-block")) {
      gpu_kernel_config.launch.threads_per_block =
          parse_positive_int(arg, require_value(i, argc, argv));
    } else if (!std::strcmp(arg, "-cuda-shmem-bytes-per-block")) {
      gpu_kernel_config.launch.dynamic_shared_memory_bytes =
          parse_size(arg, require_value(i, argc, argv));
    } else if (!std::strcmp(arg, "-cuda-compute-dtype")) {
      gpu_kernel_config.compute_data_type =
          parse_compute_data_type(arg, require_value(i, argc, argv));
    } else if (!std::strncmp(arg, "-cuda-", 6)) {
      throw std::runtime_error(std::string("unknown CUDASTF option: ") + arg);
    } else {
      arguments.core_arguments.emplace_back(arg);
      if (!std::strcmp(arg, "-h")) {
        arguments.help_requested = true;
      } else if (!std::strcmp(arg, "-and")) {
        arguments.gpu_kernel_configs.push_back(gpu_kernel_config);
        gpu_kernel_config = GpuKernelConfig{};
      }
    }
  }
  arguments.gpu_kernel_configs.push_back(gpu_kernel_config);

  if (arguments.run.context != "stream") {
    throw std::runtime_error("unsupported -cuda-context '" +
                             arguments.run.context +
                             "'; only the stream executor is implemented");
  }
  return arguments;
}

const char *compute_data_type_name(ComputeDataType type)
{
  switch (type) {
  case ComputeDataType::fp32:
    return "fp32";
  case ComputeDataType::fp64:
    return "fp64";
  }
  throw std::logic_error("unknown compute data type");
}

void print_backend_help()
{
  std::printf("\nCUDASTF backend options:\n");
  std::printf("  %-24s CUDA device id (default: 0)\n", "-cuda-device [INT]");
  std::printf("  %-24s context mode (currently: stream)\n",
              "-cuda-context [MODE]");
  std::printf("  %-24s number of warmup samples (default: 1)\n",
              "-cuda-warmup [INT]");
  std::printf("  %-24s number of measured samples (default: 5)\n",
              "-cuda-runs [INT]");
  std::printf("  %-24s write machine-readable results\n",
              "-cuda-json [FILE]");
  std::printf("  %-36s blocks launched by each task (default: 32)\n",
              "-cuda-blocks-per-task [INT]");
  std::printf("  %-36s threads in each block (default: 128)\n",
              "-cuda-threads-per-block [INT]");
  std::printf("  %-36s dynamic shared memory per block (default: 0)\n",
              "-cuda-shmem-bytes-per-block [INT]");
  std::printf("  %-36s compute_bound data type (default: fp32)\n",
              "-cuda-compute-dtype [fp32|fp64]");
  std::printf("\nCUDASTF workload data:\n");
  std::printf(
      "  -output is dependency-data bytes per task; every predecessor "
      "input is read completely once.\n");
  std::printf(
      "  memory_bound requires private -scratch bytes per task and uses "
      "-sample regions.\n");
  std::printf(
      "  -imbalance deterministically scales each busy_wait, "
      "memory_bound, or compute_bound task.\n");
}
