#ifndef TASK_BENCH_CUDASTF_PLACEMENT_H
#define TASK_BENCH_CUDASTF_PLACEMENT_H

#include <vector>

struct TaskCoordinates;
struct TaskGraph;

enum class PlacementPolicy {
  block,
  cyclic,
};

struct TaskPlacement {
  std::vector<int> devices{0};
  PlacementPolicy policy = PlacementPolicy::block;

  int device_for(const TaskGraph &task_graph,
                 const TaskCoordinates &coordinates) const;
};

const char *placement_policy_name(PlacementPolicy policy);

#endif
