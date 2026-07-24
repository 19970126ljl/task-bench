#ifndef TASK_BENCH_CUDASTF_WORKLOAD_H
#define TASK_BENCH_CUDASTF_WORKLOAD_H

#include <cstdint>
#include <stdexcept>
#include <vector>

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

using TaskIterationCounts = std::vector<std::uint64_t>;

std::uint64_t task_iteration_count(const TaskGraph &task_graph,
                                   const DagTask &dag_task);
TaskIterationCounts make_task_iteration_counts(const ExpandedDag &dag);
bool uses_task_scratch(const TaskGraph &task_graph);

struct GpuKernelResources {
  int device_id = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_memory_bytes = 0;
  int max_active_blocks_per_sm = 0;
};

using KernelResourcesByDag =
    std::vector<std::vector<GpuKernelResources>>;

#endif
