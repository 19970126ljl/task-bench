#include "identity.h"

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

class StableHash {
public:
  void add_u8(std::uint8_t value)
  {
    value_ ^= value;
    value_ *= UINT64_C(1099511628211);
  }

  template <typename T>
  void add_unsigned(T value)
  {
    static_assert(std::is_unsigned<T>::value, "unsigned integer required");
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      add_u8(static_cast<std::uint8_t>(value & 0xff));
      value >>= 8;
    }
  }

  void add_i64(std::int64_t value)
  {
    add_unsigned(static_cast<std::uint64_t>(value));
  }

  void add_u64(std::uint64_t value) { add_unsigned(value); }

  void add_f64(double value)
  {
    static_assert(sizeof(value) == sizeof(std::uint64_t),
                  "double must have a 64-bit representation");
    std::uint64_t bits = 0;
    if (value == 0.0) {
      value = 0.0;
    }
    std::memcpy(&bits, &value, sizeof(bits));
    add_u64(bits);
  }

  void add_string(const std::string &value)
  {
    add_u64(value.size());
    for (unsigned char byte : value) add_u8(byte);
  }

  std::string hex_digest() const
  {
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << value_;
    return out.str();
  }

private:
  std::uint64_t value_ = UINT64_C(14695981039346656037);
};

const char *kernel_type_name(KernelType type)
{
  switch (type) {
  case KernelType::EMPTY:
    return "empty";
  case KernelType::BUSY_WAIT:
    return "busy_wait";
  case KernelType::MEMORY_BOUND:
    return "memory_bound";
  case KernelType::COMPUTE_BOUND:
    return "compute_bound";
  default:
    throw std::logic_error("unsupported kernel type in execution identity");
  }
}

void add_effective_kernel_config(StableHash &hash,
                                 const TaskGraph &task_graph,
                                 const GpuKernelConfig &gpu_kernel_config)
{
  hash.add_string(kernel_type_name(task_graph.kernel.type));
  switch (task_graph.kernel.type) {
  case KernelType::EMPTY:
    break;
  case KernelType::BUSY_WAIT:
    hash.add_i64(task_graph.kernel.iterations);
    break;
  case KernelType::MEMORY_BOUND:
    hash.add_i64(task_graph.kernel.iterations);
    hash.add_i64(task_graph.kernel.samples);
    break;
  case KernelType::COMPUTE_BOUND:
    hash.add_i64(task_graph.kernel.iterations);
    hash.add_string(
        compute_data_type_name(gpu_kernel_config.compute_data_type));
    break;
  default:
    throw std::logic_error("unsupported kernel type in execution identity");
  }
  hash.add_f64(task_graph.kernel.imbalance);
}

}  // namespace

std::string compute_topology_hash(const std::vector<DagTask> &tasks)
{
  StableHash hash;
  hash.add_string("task-bench-dag-topology");
  hash.add_u64(tasks.size());
  for (const DagTask &task : tasks) {
    hash.add_i64(task.coordinates.timestep);
    hash.add_i64(task.coordinates.point);
    hash.add_u64(task.predecessors.size());
    for (const Predecessor &predecessor : task.predecessors) {
      hash.add_i64(predecessor.timestep);
      hash.add_i64(predecessor.point);
    }
  }
  return hash.hex_digest();
}

std::string compute_execution_config_hash(
    const RunConfig &run_config,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<ExpandedDag> &expanded_dags)
{
  if (gpu_kernel_configs.size() != expanded_dags.size()) {
    throw std::logic_error(
        "GPU kernel configuration count does not match DAG count");
  }

  StableHash hash;
  hash.add_string("cudastf-execution-config");
  hash.add_string(run_config.context);
  hash.add_string(run_config.logical_data_allocator);
  if (run_config.placement.devices.empty()) {
    throw std::logic_error("execution configuration has no CUDA devices");
  }
  hash.add_i64(run_config.placement.devices.front());
  if (run_config.placement.devices.size() > 1) {
    hash.add_string("multi-gpu-placement");
    hash.add_u64(run_config.placement.devices.size());
    for (int device : run_config.placement.devices) {
      hash.add_i64(device);
    }
    hash.add_string(placement_policy_name(run_config.placement.policy));
  }
  hash.add_u64(expanded_dags.size());

  for (std::size_t i = 0; i < expanded_dags.size(); ++i) {
    const ExpandedDag &expanded_dag = expanded_dags[i];
    const TaskGraph &task_graph = expanded_dag.task_graph;
    const GpuKernelConfig &gpu_kernel_config = gpu_kernel_configs[i];
    hash.add_string(expanded_dag.topology_hash);
    add_effective_kernel_config(
        hash, task_graph, gpu_kernel_config);
    hash.add_u64(task_graph.output_bytes_per_task);
    hash.add_u64(task_graph.scratch_bytes_per_task);
    hash.add_i64(task_graph.nb_fields);
    hash.add_i64(gpu_kernel_config.launch.blocks_per_task);
    hash.add_i64(gpu_kernel_config.launch.threads_per_block);
    hash.add_u64(
        gpu_kernel_config.launch.dynamic_shared_memory_bytes);
  }

  return hash.hex_digest();
}
