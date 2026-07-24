#ifndef TASK_BENCH_CUDASTF_WORKLOAD_H
#define TASK_BENCH_CUDASTF_WORKLOAD_H

#include <cstdint>
#include <stdexcept>

#include "arguments.h"
#include "core.h"
#include "expanded_dag.h"

enum class GpuWorkload {
  empty,
  busy_wait,
  memory_bound,
  compute_bound,
};

inline GpuWorkload gpu_workload(KernelType type)
{
  switch (type) {
  case KernelType::EMPTY:
    return GpuWorkload::empty;
  case KernelType::BUSY_WAIT:
    return GpuWorkload::busy_wait;
  case KernelType::MEMORY_BOUND:
    return GpuWorkload::memory_bound;
  case KernelType::COMPUTE_BOUND:
    return GpuWorkload::compute_bound;
  default:
    throw std::logic_error("unsupported GPU workload");
  }
}

inline const char *gpu_workload_name(KernelType type)
{
  switch (gpu_workload(type)) {
  case GpuWorkload::empty:
    return "empty";
  case GpuWorkload::busy_wait:
    return "busy_wait";
  case GpuWorkload::memory_bound:
    return "memory_bound";
  case GpuWorkload::compute_bound:
    return "compute_bound";
  }
  throw std::logic_error("unknown GPU workload");
}

std::uint64_t workload_iterations_per_task(const TaskGraph &task_graph);
bool uses_task_scratch(const TaskGraph &task_graph);

struct WorkloadModel {
  std::uint64_t logical_iterations = 0;
  std::uint64_t task_input_read_bytes = 0;
  std::uint64_t task_output_write_bytes = 0;
  std::uint64_t scratch_read_bytes = 0;
  std::uint64_t scratch_write_bytes = 0;
};

WorkloadModel workload_model(const ExpandedDag &dag);

#endif
