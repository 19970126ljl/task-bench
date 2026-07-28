#include "task_profile.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void validate_activity_envelope(
    bool has_gpu_activity, std::uint64_t start_ns,
    std::uint64_t end_ns, double elapsed_ms,
    std::uint64_t operation_count, const char *subject)
{
  if (has_gpu_activity) {
    if (start_ns >= end_ns || operation_count == 0) {
      throw std::runtime_error(
          std::string("CUPTI ") + subject + " has invalid GPU activity");
    }
    return;
  }
  if (start_ns != 0 || end_ns != 0 || elapsed_ms != 0.0 ||
      operation_count != 0) {
    throw std::runtime_error(
        std::string("CUPTI ") + subject +
        " without GPU activity has nonzero timing data");
  }
}

std::uint64_t checked_add_operations(
    std::uint64_t total, std::uint64_t value)
{
  if (value > std::numeric_limits<std::uint64_t>::max() - total) {
    throw std::runtime_error("CUPTI operation count overflows uint64_t");
  }
  return total + value;
}

}  // namespace

TaskProfileSample associate_task_profile(
    const TaskActivitySample &activity,
    const ExpectedProfileTasks &expected_tasks,
    CudaFeatureState expected_serialization)
{
  if (activity.contexts.size() != 1) {
    throw std::runtime_error(
        "each task-bench sample must produce exactly one profiled context");
  }
  const TaskProfileContext &context = activity.contexts.front();
  if (context.task_serialization != expected_serialization) {
    throw std::runtime_error(
        "CUPTI profile task serialization metadata mismatch");
  }
  validate_activity_envelope(
      context.has_gpu_activity, context.start_ns, context.end_ns,
      context.elapsed_ms, context.operation_count, "context");
  for (const TaskProfileRegion &region : context.regions) {
    validate_activity_envelope(
        region.has_gpu_activity, region.start_ns, region.end_ns,
        region.elapsed_ms, region.operation_count, "region");
  }
  if (activity.tasks.size() != expected_tasks.size()) {
    throw std::runtime_error(
        "CUPTI profile task count does not match submitted task count");
  }
  if (context.task_count != expected_tasks.size()) {
    throw std::runtime_error(
        "CUPTI context task count does not match submitted task count");
  }
  for (const auto &entry : expected_tasks) {
    if (entry.first.first != context.context_id) {
      throw std::runtime_error("CUPTI profile context id mismatch");
    }
  }

  TaskProfileSample result;
  result.cupti_timestamp_origin_ns =
      activity.cupti_timestamp_origin_ns;
  result.contexts = activity.contexts;
  result.tasks.reserve(activity.tasks.size());
  std::set<RuntimeProfileKey> matched;
  std::uint64_t task_operation_count = 0;
  for (const TaskActivityRecord &task : activity.tasks) {
    if (task.context_id != context.context_id) {
      throw std::runtime_error("CUPTI profile task context id mismatch");
    }
    const RuntimeProfileKey runtime_key{task.context_id, task.task_id};
    const auto expected = expected_tasks.find(runtime_key);
    if (expected == expected_tasks.end()) {
      throw std::runtime_error(
          "CUPTI profile contains an unknown context/task id");
    }
    if (!matched.insert(runtime_key).second) {
      throw std::runtime_error(
          "CUPTI profile contains a duplicate context/task id");
    }
    if (task.symbol != expected->second.symbol) {
      throw std::runtime_error("CUPTI profile task symbol mismatch");
    }
    validate_activity_envelope(
        task.has_gpu_activity, task.start_ns, task.end_ns,
        task.elapsed_ms, task.operation_count, "task");

    std::uint64_t device_operation_count = 0;
    for (const TaskDeviceProfile &device : task.device_timings) {
      if (device.device_id != expected->second.configured_device) {
        throw std::runtime_error(
            "CUPTI profile device does not match configured placement");
      }
      validate_activity_envelope(
          true, device.start_ns, device.end_ns, device.elapsed_ms,
          device.operation_count, "task device timing");
      device_operation_count = checked_add_operations(
          device_operation_count, device.operation_count);
    }
    if (task.has_gpu_activity) {
      if (task.device_timings.size() != 1 ||
          device_operation_count != task.operation_count) {
        throw std::runtime_error("CUPTI task has invalid GPU activity");
      }
    } else if (!task.device_timings.empty()) {
      throw std::runtime_error(
          "CUPTI task without GPU activity has device timings");
    }
    task_operation_count = checked_add_operations(
        task_operation_count, task.operation_count);

    TaskProfileRecord converted;
    converted.key = expected->second.key;
    converted.configured_device = expected->second.configured_device;
    converted.context_id = task.context_id;
    converted.region_id = task.region_id;
    converted.task_id = task.task_id;
    converted.symbol = task.symbol;
    converted.has_gpu_activity = task.has_gpu_activity;
    converted.start_ns = task.start_ns;
    converted.end_ns = task.end_ns;
    converted.elapsed_ms = task.elapsed_ms;
    converted.operation_count = task.operation_count;
    converted.device_timings = task.device_timings;
    result.tasks.push_back(std::move(converted));
  }

  if (matched.size() != expected_tasks.size()) {
    throw std::runtime_error("CUPTI profile is missing submitted tasks");
  }
  if (task_operation_count != context.operation_count) {
    throw std::runtime_error(
        "CUPTI context operation count does not match its tasks");
  }
  if (expected_serialization == CudaFeatureState::enabled) {
    std::vector<const TaskProfileRecord *> active_tasks;
    for (const TaskProfileRecord &task : result.tasks) {
      if (task.has_gpu_activity) active_tasks.push_back(&task);
    }
    std::sort(
        active_tasks.begin(), active_tasks.end(),
        [](const TaskProfileRecord *lhs, const TaskProfileRecord *rhs) {
          if (lhs->start_ns != rhs->start_ns) {
            return lhs->start_ns < rhs->start_ns;
          }
          return lhs->end_ns < rhs->end_ns;
        });
    for (std::size_t i = 1; i < active_tasks.size(); ++i) {
      if (active_tasks[i - 1]->end_ns > active_tasks[i]->start_ns) {
        throw std::runtime_error(
            "serialized CUDASTF task GPU activity overlaps");
      }
    }
  }
  return result;
}
