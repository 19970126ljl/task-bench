#ifndef TASK_BENCH_CUDASTF_EXPANDED_DAG_H
#define TASK_BENCH_CUDASTF_EXPANDED_DAG_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core.h"

constexpr std::size_t kMaxTaskInputs = 32;
constexpr std::size_t kMaxTaskDataAccesses = kMaxTaskInputs + 1;

struct TaskCoordinates {
  std::int64_t dag_index;
  std::int64_t timestep;
  std::int64_t point;
};

struct Predecessor {
  std::int64_t timestep;
  std::int64_t point;
};

struct DagTask {
  TaskCoordinates coordinates;
  std::vector<Predecessor> predecessors;
};

struct ExpandedDag {
  TaskGraph task_graph;
  std::vector<DagTask> tasks;
  std::uint64_t dependency_edges = 0;
  std::uint64_t task_data_accesses = 0;
  std::size_t task_data_store_bytes = 0;
  std::string topology_hash;

  std::int64_t dag_index() const { return task_graph.graph_index; }
};

std::vector<ExpandedDag> expand_task_graphs(const App &task_bench_app);

#endif
