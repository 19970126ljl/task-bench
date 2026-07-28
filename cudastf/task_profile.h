#ifndef TASK_BENCH_CUDASTF_TASK_PROFILE_H
#define TASK_BENCH_CUDASTF_TASK_PROFILE_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "arguments.h"

struct DagTaskKey {
  std::int64_t dag_index = 0;
  std::int64_t timestep = 0;
  std::int64_t point = 0;
};

inline bool operator==(const DagTaskKey &lhs,
                       const DagTaskKey &rhs)
{
  return lhs.dag_index == rhs.dag_index &&
         lhs.timestep == rhs.timestep && lhs.point == rhs.point;
}

inline bool operator<(const DagTaskKey &lhs,
                      const DagTaskKey &rhs)
{
  if (lhs.dag_index != rhs.dag_index) {
    return lhs.dag_index < rhs.dag_index;
  }
  if (lhs.timestep != rhs.timestep) return lhs.timestep < rhs.timestep;
  return lhs.point < rhs.point;
}

struct TaskDeviceProfile {
  int device_id = 0;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  double elapsed_ms = 0.0;
  std::uint64_t operation_count = 0;
};

struct TaskProfileRecord {
  DagTaskKey key;
  int configured_device = 0;
  std::uint64_t context_id = 0;
  std::uint64_t region_id = 0;
  int task_id = 0;
  std::string symbol;
  bool has_gpu_activity = false;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  double elapsed_ms = 0.0;
  std::uint64_t operation_count = 0;
  std::vector<TaskDeviceProfile> device_timings;
};

struct TaskProfileRegion {
  std::uint64_t region_id = 0;
  std::string label;
  bool has_gpu_activity = false;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  double elapsed_ms = 0.0;
  std::uint64_t task_count = 0;
  std::uint64_t operation_count = 0;
};

struct TaskProfileContext {
  std::uint64_t context_id = 0;
  std::string label;
  bool has_gpu_activity = false;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  double elapsed_ms = 0.0;
  std::uint64_t task_count = 0;
  std::uint64_t operation_count = 0;
  CudaFeatureState task_serialization = CudaFeatureState::disabled;
  std::vector<TaskProfileRegion> regions;
};

struct TaskProfileSample {
  std::uint64_t cupti_timestamp_origin_ns = 0;
  std::vector<TaskProfileContext> contexts;
  std::vector<TaskProfileRecord> tasks;
};

struct TaskActivityRecord {
  std::uint64_t context_id = 0;
  std::uint64_t region_id = 0;
  int task_id = 0;
  std::string symbol;
  bool has_gpu_activity = false;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  double elapsed_ms = 0.0;
  std::uint64_t operation_count = 0;
  std::vector<TaskDeviceProfile> device_timings;
};

struct TaskActivitySample {
  std::uint64_t cupti_timestamp_origin_ns = 0;
  std::vector<TaskProfileContext> contexts;
  std::vector<TaskActivityRecord> tasks;
};

struct ExpectedProfileTask {
  DagTaskKey key;
  int configured_device = 0;
  std::string symbol;
};

using RuntimeProfileKey = std::pair<std::uint64_t, int>;
using ExpectedProfileTasks =
    std::map<RuntimeProfileKey, ExpectedProfileTask>;

TaskProfileSample associate_task_profile(
    const TaskActivitySample &activity,
    const ExpectedProfileTasks &expected_tasks,
    CudaFeatureState expected_serialization);

#endif
