#include "workload.h"

#include "core_kernel.h"

std::uint64_t task_iteration_count(const TaskGraph &task_graph,
                                   const DagTask &dag_task)
{
  switch (gpu_workload(task_graph.kernel.type)) {
  case GpuWorkload::empty:
    return 0;
  case GpuWorkload::busy_wait:
  case GpuWorkload::memory_bound:
  case GpuWorkload::compute_bound: {
    if (task_graph.kernel.imbalance == 0.0) {
      return static_cast<std::uint64_t>(
          task_graph.kernel.iterations);
    }
    const long iterations = select_imbalance_iterations(
        task_graph.kernel, task_graph.graph_index,
        dag_task.coordinates.timestep, dag_task.coordinates.point);
    if (iterations < 0) {
      throw std::logic_error("task iteration count is negative");
    }
    return static_cast<std::uint64_t>(iterations);
  }
  }
  throw std::logic_error("unknown GPU workload");
}

bool uses_task_scratch(const TaskGraph &task_graph)
{
  return gpu_workload(task_graph.kernel.type) ==
         GpuWorkload::memory_bound;
}

TaskIterationCounts make_task_iteration_counts(const ExpandedDag &dag)
{
  TaskIterationCounts result;
  result.reserve(dag.tasks.size());
  for (const DagTask &dag_task : dag.tasks) {
    result.push_back(task_iteration_count(dag.task_graph, dag_task));
  }
  return result;
}
