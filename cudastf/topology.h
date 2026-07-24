#ifndef TASK_BENCH_CUDASTF_TOPOLOGY_H
#define TASK_BENCH_CUDASTF_TOPOLOGY_H

#include <cstdint>
#include <vector>

#include "expanded_dag.h"

struct DagTopologyMetrics {
  std::uint64_t tasks = 0;
  std::uint64_t dependency_edges = 0;
  std::uint64_t critical_path_length = 0;
  double average_parallelism = 0.0;
  std::uint64_t peak_parallelism = 0;
  double parallelism_p50 = 0.0;
  double parallelism_p95 = 0.0;
  double parallelism_cv = 0.0;
};

struct TopologyMetrics {
  std::vector<DagTopologyMetrics> dags;
  // Disjoint-union metrics formed by stacking per-DAG ASAP level widths.
  DagTopologyMetrics combined;
};

TopologyMetrics compute_topology_metrics(
    const std::vector<ExpandedDag> &dags);

#endif
