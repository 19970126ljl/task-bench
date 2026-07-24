#include "expanded_dag.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "identity.h"

namespace {

std::size_t checked_task_data_store_bytes(const TaskGraph &task_graph)
{
  const std::uint64_t fields =
      static_cast<std::uint64_t>(task_graph.nb_fields);
  const std::uint64_t width =
      static_cast<std::uint64_t>(task_graph.max_width);
  const std::uint64_t bytes =
      static_cast<std::uint64_t>(task_graph.output_bytes_per_task);
  const std::uint64_t limit = std::numeric_limits<std::size_t>::max();

  if (fields != 0 && width > limit / fields) {
    throw std::runtime_error("task data store size overflows size_t");
  }
  const std::uint64_t slots = fields * width;
  if (slots != 0 && bytes > limit / slots) {
    throw std::runtime_error("task data store size overflows size_t");
  }
  return static_cast<std::size_t>(slots * bytes);
}

bool is_supported_kernel(KernelType type)
{
  return type == KernelType::EMPTY ||
         type == KernelType::BUSY_WAIT ||
         type == KernelType::MEMORY_BOUND ||
         type == KernelType::COMPUTE_BOUND;
}

}  // namespace

std::vector<ExpandedDag> expand_task_graphs(const App &task_bench_app)
{
  std::vector<ExpandedDag> expanded_dags;
  expanded_dags.reserve(task_bench_app.graphs.size());

  for (const TaskGraph &task_graph : task_bench_app.graphs) {
    if (!is_supported_kernel(task_graph.kernel.type)) {
      throw std::runtime_error(
          "CUDASTF supports '-kernel empty', 'busy_wait', 'memory_bound', "
          "and 'compute_bound'");
    }
    if (task_graph.kernel.type == KernelType::MEMORY_BOUND) {
      if (task_graph.scratch_bytes_per_task == 0) {
        throw std::runtime_error(
            "CUDASTF memory_bound requires '-scratch' greater than zero");
      }
      if (task_graph.kernel.samples <= 0) {
        throw std::runtime_error(
            "CUDASTF memory_bound requires '-sample' greater than zero");
      }
      const std::size_t samples =
          static_cast<std::size_t>(task_graph.kernel.samples);
      if (task_graph.scratch_bytes_per_task % samples != 0) {
        throw std::runtime_error(
            "CUDASTF memory_bound requires '-scratch' to be divisible by "
            "'-sample'");
      }
      const std::size_t sample_bytes =
          task_graph.scratch_bytes_per_task / samples;
      if (sample_bytes < 2 || sample_bytes % 2 != 0) {
        throw std::runtime_error(
            "CUDASTF memory_bound requires each scratch sample to contain "
            "equal source and destination regions");
      }
    } else if (task_graph.scratch_bytes_per_task != 0) {
      throw std::runtime_error(
          "CUDASTF uses '-scratch' only with '-kernel memory_bound'");
    }

    ExpandedDag expanded_dag;
    expanded_dag.task_graph = task_graph;
    expanded_dag.task_data_store_bytes =
        checked_task_data_store_bytes(task_graph);

    for (long timestep = 0; timestep < task_graph.timesteps; ++timestep) {
      const long offset = task_graph.offset_at_timestep(timestep);
      const long width = task_graph.width_at_timestep(timestep);
      const long previous_offset =
          task_graph.offset_at_timestep(timestep - 1);
      const long previous_width =
          task_graph.width_at_timestep(timestep - 1);
      const long previous_end = previous_offset + previous_width;
      const long dependence_set =
          task_graph.dependence_set_at_timestep(timestep);

      for (long point = offset; point < offset + width; ++point) {
        DagTask dag_task{
            {task_graph.graph_index, timestep, point}, {}};
        const auto intervals =
            task_graph.dependencies(dependence_set, point);
        for (const auto &interval : intervals) {
          const long first = std::max(interval.first, previous_offset);
          const long last = std::min(interval.second, previous_end - 1);
          for (long predecessor_point = first;
               predecessor_point <= last; ++predecessor_point) {
            dag_task.predecessors.push_back(
                {timestep - 1, predecessor_point});
          }
        }

        if (task_graph.nb_fields == 1) {
          const auto cross_point =
              std::find_if(dag_task.predecessors.begin(),
                           dag_task.predecessors.end(),
                           [point](const Predecessor &predecessor) {
                             return predecessor.point != point;
                           });
          if (cross_point != dag_task.predecessors.end()) {
            std::ostringstream message;
            message
                << "unsupported field reuse: task (dag="
                << task_graph.graph_index << ",timestep=" << timestep
                << ",point=" << point << ") reads predecessor point "
                << cross_point->point
                << " while '-field 1' aliases current and previous "
                   "timesteps; use '-field 2' or more";
            throw std::runtime_error(message.str());
          }
        }

        if (dag_task.predecessors.size() > kMaxTaskInputs) {
          std::ostringstream message;
          message << "unsupported DAG instance: task (dag="
                  << task_graph.graph_index << ",timestep=" << timestep
                  << ",point=" << point << ") has "
                  << dag_task.predecessors.size()
                  << " predecessors; the bounded fan-in implementation "
                     "supports at most "
                  << kMaxTaskInputs << " predecessors ("
                  << kMaxTaskDataAccesses
                  << " task-data accesses including output)";
          throw std::runtime_error(message.str());
        }

        expanded_dag.max_fanin =
            std::max(expanded_dag.max_fanin,
                     dag_task.predecessors.size());
        if (expanded_dag.dependency_edges >
            std::numeric_limits<std::uint64_t>::max() -
                dag_task.predecessors.size()) {
          throw std::runtime_error(
              "dependency edge count overflows uint64_t");
        }
        expanded_dag.dependency_edges +=
            dag_task.predecessors.size();

        expanded_dag.tasks.push_back(std::move(dag_task));
      }
    }

    if (expanded_dag.tasks.size() >
        std::numeric_limits<std::uint64_t>::max() -
            expanded_dag.dependency_edges) {
      throw std::runtime_error(
          "task data access count overflows uint64_t");
    }
    expanded_dag.task_data_accesses =
        expanded_dag.tasks.size() + expanded_dag.dependency_edges;
    expanded_dag.topology_hash =
        compute_topology_hash(expanded_dag.tasks);
    expanded_dags.push_back(std::move(expanded_dag));
  }
  return expanded_dags;
}
