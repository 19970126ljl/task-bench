#include "workload.h"

#include <limits>
#include <string>

namespace {

std::uint64_t checked_multiply(std::uint64_t lhs, std::uint64_t rhs,
                               const char *name)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs * rhs;
}

}  // namespace

std::uint64_t workload_iterations_per_task(const TaskGraph &task_graph)
{
  switch (gpu_workload(task_graph.kernel.type)) {
  case GpuWorkload::empty:
    return 0;
  case GpuWorkload::busy_wait:
  case GpuWorkload::memory_bound:
  case GpuWorkload::compute_bound:
    return static_cast<std::uint64_t>(task_graph.kernel.iterations);
  }
  throw std::logic_error("unknown GPU workload");
}

bool uses_task_scratch(const TaskGraph &task_graph)
{
  return gpu_workload(task_graph.kernel.type) ==
         GpuWorkload::memory_bound;
}

WorkloadModel workload_model(const ExpandedDag &dag)
{
  WorkloadModel result;
  const std::uint64_t bytes = dag.task_graph.output_bytes_per_task;
  const std::uint64_t iterations =
      workload_iterations_per_task(dag.task_graph);
  result.task_input_read_bytes =
      checked_multiply(dag.dependency_edges, bytes,
                       "task input read byte count");
  result.task_output_write_bytes =
      checked_multiply(dag.tasks.size(), bytes,
                       "task output write byte count");
  result.logical_iterations =
      checked_multiply(dag.tasks.size(), iterations,
                       "workload iteration count");

  if (uses_task_scratch(dag.task_graph)) {
    const std::uint64_t scratch_bytes =
        dag.task_graph.scratch_bytes_per_task;
    const std::uint64_t samples =
        static_cast<std::uint64_t>(dag.task_graph.kernel.samples);
    const std::uint64_t copy_bytes = scratch_bytes / samples / 2;
    result.scratch_read_bytes =
        checked_multiply(result.logical_iterations, copy_bytes,
                         "scratch read byte count");
    result.scratch_write_bytes =
        checked_multiply(result.logical_iterations, copy_bytes,
                         "scratch write byte count");
  }

  return result;
}
