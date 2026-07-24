#include "task_placement.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

#include "core.h"
#include "expanded_dag.h"

int TaskPlacement::device_for(
    const TaskGraph &task_graph,
    const TaskCoordinates &coordinates) const
{
  if (devices.empty()) {
    throw std::logic_error("task placement has no devices");
  }
  if (task_graph.max_width <= 0 ||
      coordinates.point < 0 ||
      coordinates.point >= task_graph.max_width) {
    throw std::logic_error("task point is outside the placement width");
  }

  const std::uint64_t point =
      static_cast<std::uint64_t>(coordinates.point);
  const std::uint64_t width =
      static_cast<std::uint64_t>(task_graph.max_width);
  const std::uint64_t device_count =
      static_cast<std::uint64_t>(devices.size());
  std::uint64_t index = 0;
  switch (policy) {
  case TaskPlacementPolicy::block:
    if (point != 0 &&
        device_count >
            std::numeric_limits<std::uint64_t>::max() / point) {
      throw std::logic_error("task placement calculation overflow");
    }
    index = point * device_count / width;
    break;
  case TaskPlacementPolicy::cyclic:
    index = point % device_count;
    break;
  }
  return devices.at(static_cast<std::size_t>(index));
}

const char *task_placement_policy_name(TaskPlacementPolicy policy)
{
  switch (policy) {
  case TaskPlacementPolicy::block:
    return "block";
  case TaskPlacementPolicy::cyclic:
    return "cyclic";
  }
  throw std::logic_error("unknown task placement policy");
}

const char *task_placement_policy_description(TaskPlacementPolicy policy)
{
  switch (policy) {
  case TaskPlacementPolicy::block:
    return "contiguous point ranges";
  case TaskPlacementPolicy::cyclic:
    return "round-robin by point";
  }
  throw std::logic_error("unknown task placement policy");
}
