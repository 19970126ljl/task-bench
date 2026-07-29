#include "workload.h"

#include <limits>
#include <map>
#include <set>
#include <utility>

#include "core_kernel.h"
#include "results.h"

namespace {

std::uint64_t checked_add_capacity(
    std::uint64_t left, std::uint64_t right)
{
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error("kernel capacity addition overflow");
  }
  return left + right;
}

std::uint64_t checked_multiply_capacity(
    std::uint64_t left, std::uint64_t right)
{
  if (left != 0 &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error("kernel capacity multiplication overflow");
  }
  return left * right;
}

std::uint64_t ceil_divide_capacity(
    std::uint64_t numerator, std::uint64_t denominator)
{
  if (denominator == 0) {
    throw std::logic_error("kernel capacity divisor is zero");
  }
  return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

}  // namespace

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

KernelCapacityMetrics compute_kernel_capacity(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<CudaDeviceInfo> &devices,
    const KernelResourcesByDag &kernel_resources)
{
  if (expanded_dags.size() != gpu_kernel_configs.size() ||
      expanded_dags.size() != kernel_resources.size()) {
    throw std::logic_error("kernel capacity DAG input count mismatch");
  }
  std::map<int, const CudaDeviceInfo *> devices_by_id;
  for (const CudaDeviceInfo &device : devices) {
    if (device.sm_count <= 0) {
      throw std::logic_error("CUDA device SM count is not positive");
    }
    if (!devices_by_id.emplace(device.device_id, &device).second) {
      throw std::logic_error("duplicate CUDA device id");
    }
  }

  KernelCapacityMetrics result;
  result.dags.reserve(expanded_dags.size());
  for (std::size_t dag_index = 0;
       dag_index < expanded_dags.size(); ++dag_index) {
    const ExpandedDag &dag = expanded_dags[dag_index];
    const int blocks_per_task =
        gpu_kernel_configs[dag_index].launch.blocks_per_task;
    if (blocks_per_task <= 0) {
      throw std::logic_error("blocks per task is not positive");
    }

    std::set<int> used_devices;
    for (const DagTask &task : dag.tasks) {
      used_devices.insert(run_config.task_placement.device_for(
          dag.task_graph, task.coordinates));
    }
    if (used_devices.empty()) {
      throw std::logic_error("DAG has no assigned CUDA device");
    }

    std::map<int, const GpuKernelResources *> resources_by_device;
    for (const GpuKernelResources &resources :
         kernel_resources[dag_index]) {
      if (!resources_by_device.emplace(
              resources.device_id, &resources).second) {
        throw std::logic_error("duplicate GPU kernel resource device id");
      }
    }

    DagKernelCapacity dag_capacity;
    dag_capacity.dag_index = dag.dag_index();
    for (const CudaDeviceInfo &device : devices) {
      if (used_devices.find(device.device_id) == used_devices.end()) {
        continue;
      }
      const auto resources = resources_by_device.find(device.device_id);
      if (resources == resources_by_device.end()) {
        throw std::logic_error(
            "missing GPU kernel resources for assigned device");
      }
      if (resources->second->max_active_blocks_per_sm <= 0) {
        throw std::logic_error(
            "maximum active blocks per SM is not positive");
      }
      const std::uint64_t resident_blocks = checked_multiply_capacity(
          static_cast<std::uint64_t>(device.sm_count),
          static_cast<std::uint64_t>(
              resources->second->max_active_blocks_per_sm));
      const std::uint64_t saturation_tasks = ceil_divide_capacity(
          resident_blocks, static_cast<std::uint64_t>(blocks_per_task));
      dag_capacity.devices.push_back(
          {device.device_id, device.sm_count,
           resources->second->max_active_blocks_per_sm,
           resident_blocks, saturation_tasks});
      dag_capacity.combined.resident_blocks = checked_add_capacity(
          dag_capacity.combined.resident_blocks, resident_blocks);
      dag_capacity.combined.occupancy_saturation_tasks =
          checked_add_capacity(
              dag_capacity.combined.occupancy_saturation_tasks,
              saturation_tasks);
    }
    if (dag_capacity.devices.size() != used_devices.size()) {
      throw std::logic_error("assigned CUDA device was not inspected");
    }
    dag_capacity.combined.device_count = dag_capacity.devices.size();
    result.dags.push_back(std::move(dag_capacity));
  }
  return result;
}
