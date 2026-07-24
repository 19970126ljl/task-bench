#include "topology.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct DagTaskKey {
  std::int64_t timestep;
  std::int64_t point;

  bool operator==(const DagTaskKey &other) const
  {
    return timestep == other.timestep && point == other.point;
  }
};

struct DagTaskKeyHash {
  std::size_t operator()(const DagTaskKey &task_key) const
  {
    const std::size_t timestep =
        std::hash<std::int64_t>{}(task_key.timestep);
    const std::size_t point =
        std::hash<std::int64_t>{}(task_key.point);
    return timestep ^ (point + 0x9e3779b9U + (timestep << 6) +
                       (timestep >> 2));
  }
};

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const char *name)
{
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    throw std::runtime_error(std::string(name) + " overflows uint64_t");
  }
  return lhs + rhs;
}

DagTopologyMetrics summarize_level_widths(
    std::uint64_t tasks, std::uint64_t dependency_edges,
    const std::vector<std::uint64_t> &level_widths)
{
  if (tasks == 0 || level_widths.empty()) {
    throw std::logic_error("cannot summarize an empty DAG topology");
  }

  DagTopologyMetrics result;
  result.tasks = tasks;
  result.dependency_edges = dependency_edges;
  result.critical_path_length = level_widths.size();
  const long double average_parallelism =
      static_cast<long double>(tasks) /
      static_cast<long double>(result.critical_path_length);
  result.average_parallelism =
      static_cast<double>(average_parallelism);
  result.peak_parallelism =
      *std::max_element(level_widths.begin(), level_widths.end());

  std::vector<std::uint64_t> sorted_widths = level_widths;
  std::sort(sorted_widths.begin(), sorted_widths.end());
  const std::size_t middle = sorted_widths.size() / 2;
  if (sorted_widths.size() % 2 == 0) {
    result.parallelism_p50 = static_cast<double>(
        (static_cast<long double>(sorted_widths[middle - 1]) +
         static_cast<long double>(sorted_widths[middle])) /
        2.0L);
  } else {
    result.parallelism_p50 =
        static_cast<double>(sorted_widths[middle]);
  }

  // ceil(19 * N / 20) without floating point or multiplication overflow.
  const std::size_t p95_rank =
      sorted_widths.size() - sorted_widths.size() / 20;
  const std::size_t p95_index = p95_rank - 1;
  result.parallelism_p95 =
      static_cast<double>(sorted_widths[p95_index]);

  long double squared_difference_sum = 0.0L;
  for (std::uint64_t width : level_widths) {
    const long double difference =
        static_cast<long double>(width) - average_parallelism;
    squared_difference_sum += difference * difference;
  }
  const long double standard_deviation =
      std::sqrt(squared_difference_sum /
                static_cast<long double>(level_widths.size()));
  result.parallelism_cv = static_cast<double>(
      standard_deviation / average_parallelism);
  return result;
}

struct DagAnalysis {
  DagTopologyMetrics metrics;
  std::vector<std::uint64_t> level_widths;
};

DagAnalysis analyze_dag(const ExpandedDag &dag)
{
  if (dag.tasks.empty()) {
    throw std::logic_error("cannot analyze an empty DAG topology");
  }
  if (dag.tasks.size() > std::numeric_limits<std::uint64_t>::max()) {
    throw std::runtime_error("task count overflows uint64_t");
  }

  std::unordered_map<DagTaskKey, std::uint64_t, DagTaskKeyHash>
      finish_level_by_task;
  finish_level_by_task.reserve(dag.tasks.size());
  std::vector<std::uint64_t> level_widths;
  std::uint64_t dependency_edges = 0;

  for (const DagTask &task : dag.tasks) {
    const DagTaskKey task_key{
        task.coordinates.timestep, task.coordinates.point};
    if (finish_level_by_task.find(task_key) !=
        finish_level_by_task.end()) {
      throw std::logic_error("DAG topology contains a duplicate task");
    }

    std::uint64_t start_level = 0;
    for (const Predecessor &predecessor : task.predecessors) {
      dependency_edges =
          checked_add(dependency_edges, 1, "dependency edge count");
      const auto predecessor_finish_level = finish_level_by_task.find(
          {predecessor.timestep, predecessor.point});
      if (predecessor_finish_level == finish_level_by_task.end()) {
        throw std::logic_error(
            "DAG predecessor is missing or appears after its successor");
      }
      start_level =
          std::max(start_level, predecessor_finish_level->second);
    }

    if (start_level > std::numeric_limits<std::size_t>::max() - 1) {
      throw std::runtime_error("critical path length overflows size_t");
    }
    const std::size_t level_index =
        static_cast<std::size_t>(start_level);
    if (level_widths.size() <= level_index) {
      level_widths.resize(level_index + 1, 0);
    }
    level_widths[level_index] = checked_add(
        level_widths[level_index], 1, "ASAP level width");
    finish_level_by_task.emplace(task_key, start_level + 1);
  }

  if (dependency_edges != dag.dependency_edges) {
    throw std::logic_error(
        "DAG dependency edge count does not match its task records");
  }

  const std::uint64_t tasks =
      static_cast<std::uint64_t>(dag.tasks.size());
  return {summarize_level_widths(
              tasks, dependency_edges, level_widths),
          std::move(level_widths)};
}

}  // namespace

TopologyMetrics compute_topology_metrics(
    const std::vector<ExpandedDag> &dags)
{
  if (dags.empty()) {
    throw std::logic_error("cannot analyze an empty DAG set");
  }

  TopologyMetrics result;
  result.dags.reserve(dags.size());
  std::vector<std::uint64_t> combined_level_widths;
  std::uint64_t combined_tasks = 0;
  std::uint64_t combined_edges = 0;

  for (const ExpandedDag &dag : dags) {
    DagAnalysis analysis = analyze_dag(dag);
    combined_tasks = checked_add(
        combined_tasks, analysis.metrics.tasks, "combined task count");
    combined_edges = checked_add(
        combined_edges, analysis.metrics.dependency_edges,
        "combined dependency edge count");
    if (combined_level_widths.size() <
        analysis.level_widths.size()) {
      combined_level_widths.resize(analysis.level_widths.size(), 0);
    }
    for (std::size_t level = 0;
         level < analysis.level_widths.size(); ++level) {
      combined_level_widths[level] = checked_add(
          combined_level_widths[level], analysis.level_widths[level],
          "combined ASAP level width");
    }
    result.dags.push_back(analysis.metrics);
  }

  result.combined = summarize_level_widths(
      combined_tasks, combined_edges, combined_level_widths);
  return result;
}
