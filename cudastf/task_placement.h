#ifndef TASK_BENCH_CUDASTF_TASK_PLACEMENT_H
#define TASK_BENCH_CUDASTF_TASK_PLACEMENT_H

#include <vector>

struct TaskCoordinates;
struct TaskGraph;

enum class TaskPlacementPolicy {
  block,
  cyclic,
};

struct TaskPlacement {
  std::vector<int> devices{0};
  TaskPlacementPolicy policy = TaskPlacementPolicy::block;

  int device_for(const TaskGraph &task_graph,
                 const TaskCoordinates &coordinates) const;
};

const char *task_placement_policy_name(TaskPlacementPolicy policy);
const char *task_placement_policy_description(TaskPlacementPolicy policy);

#endif
