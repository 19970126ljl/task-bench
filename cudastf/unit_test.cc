#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arguments.h"
#include "core.h"
#include "data_access.h"
#include "expanded_dag.h"
#include "identity.h"
#include "results.h"
#include "workload.h"

namespace {

Arguments parse_backend_arguments(std::vector<std::string> arguments)
{
  arguments.insert(arguments.begin(), "unit_test");
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments) {
    argv.push_back(argument.data());
  }
  return parse_arguments(static_cast<int>(argv.size()), argv.data());
}

ExpandedDag build_dag(std::vector<std::string> arguments)
{
  arguments.insert(arguments.begin(), "unit_test");
  std::vector<char *> argv;
  argv.reserve(arguments.size());
  for (std::string &argument : arguments) {
    argv.push_back(argument.data());
  }
  App task_bench_app(static_cast<int>(argv.size()), argv.data());
  std::vector<ExpandedDag> expanded_dags =
      expand_task_graphs(task_bench_app);
  if (expanded_dags.size() != 1) {
    throw std::runtime_error("expected exactly one expanded DAG");
  }
  return std::move(expanded_dags[0]);
}

struct PreparedRun {
  Arguments arguments;
  std::vector<ExpandedDag> expanded_dags;
};

PreparedRun prepare_run(std::vector<std::string> arguments)
{
  Arguments parsed = parse_backend_arguments(std::move(arguments));
  std::vector<char *> core_argv = parsed.core_argv();
  App task_bench_app(
      static_cast<int>(core_argv.size()), core_argv.data());
  std::vector<ExpandedDag> expanded_dags =
      expand_task_graphs(task_bench_app);
  if (parsed.gpu_kernel_configs.size() != expanded_dags.size()) {
    throw std::runtime_error("test GPU configuration count mismatch");
  }
  return {std::move(parsed), std::move(expanded_dags)};
}

const DagTask &find_dag_task(
    const ExpandedDag &expanded_dag, long timestep, long point)
{
  for (const DagTask &dag_task : expanded_dag.tasks) {
    if (dag_task.coordinates.timestep == timestep &&
        dag_task.coordinates.point == point) {
      return dag_task;
    }
  }
  throw std::runtime_error("golden task was not generated");
}

void expect_counts(const char *name, const ExpandedDag &expanded_dag,
                   std::uint64_t tasks, std::uint64_t edges)
{
  if (expanded_dag.tasks.size() != tasks ||
      expanded_dag.dependency_edges != edges) {
    std::ostringstream error;
    error << name << " count mismatch: expected (" << tasks << "," << edges
          << "), got ("
          << expanded_dag.tasks.size() << ","
          << expanded_dag.dependency_edges << ")";
    throw std::runtime_error(error.str());
  }
}

void expect_predecessors(const char *name, const ExpandedDag &expanded_dag,
                         long timestep, long point,
                         const std::vector<long> &expected_points)
{
  const DagTask &dag_task =
      find_dag_task(expanded_dag, timestep, point);
  if (dag_task.predecessors.size() != expected_points.size()) {
    throw std::runtime_error(std::string(name) +
                             " predecessor count mismatch");
  }
  for (std::size_t i = 0; i < expected_points.size(); ++i) {
    if (dag_task.predecessors[i].timestep != timestep - 1 ||
        dag_task.predecessors[i].point != expected_points[i]) {
      std::ostringstream error;
      error << name << " predecessor mismatch at index " << i
            << ": expected (" << timestep - 1 << "," << expected_points[i]
            << "), got (" << dag_task.predecessors[i].timestep << ","
            << dag_task.predecessors[i].point << ")";
      throw std::runtime_error(error.str());
    }
  }
}

void expect_near(const char *name, double actual, double expected)
{
  if (std::fabs(actual - expected) > 1e-12) {
    std::ostringstream error;
    error << name << ": expected " << expected << ", got " << actual;
    throw std::runtime_error(error.str());
  }
}

void expect_data_access(const char *name, const DataAccess &actual,
                        std::int64_t dag_index, std::int64_t field,
                        std::int64_t point, DataAccessMode mode)
{
  if (actual.data.dag_index != dag_index ||
      actual.data.field != field ||
      actual.data.point != point ||
      actual.mode != mode) {
    std::ostringstream error;
    error << name << ": unexpected data access (dag="
          << actual.data.dag_index << ",field=" << actual.data.field
          << ",point=" << actual.data.point << ")";
    throw std::runtime_error(error.str());
  }
}

void test_expanded_dag_goldens()
{
  const ExpandedDag trivial =
      build_dag({"-steps", "4", "-width", "5", "-type", "trivial"});
  const ExpandedDag no_comm =
      build_dag({"-steps", "4", "-width", "5", "-type", "no_comm"});
  const ExpandedDag stencil =
      build_dag({"-steps", "4", "-width", "5", "-type", "stencil_1d"});
  const ExpandedDag periodic = build_dag(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d_periodic"});
  const ExpandedDag dom =
      build_dag({"-steps", "7", "-width", "4", "-type", "dom"});
  const ExpandedDag tree =
      build_dag({"-steps", "5", "-width", "8", "-type", "tree"});
  const ExpandedDag fft =
      build_dag({"-steps", "4", "-width", "8", "-type", "fft"});
  const ExpandedDag all_to_all =
      build_dag({"-steps", "3", "-width", "4", "-type", "all_to_all"});
  const ExpandedDag nearest = build_dag(
      {"-steps", "4", "-width", "5", "-type", "nearest", "-radix", "3"});
  const ExpandedDag spread =
      build_dag({"-steps", "4", "-width", "8", "-type", "spread",
                 "-radix", "4", "-period", "2"});
  const ExpandedDag random_nearest = build_dag(
      {"-steps", "4", "-width", "8", "-type", "random_nearest",
       "-radix", "5", "-period", "3", "-fraction", "0.5"});

  expect_counts("trivial", trivial, 20, 0);
  expect_counts("no_comm", no_comm, 20, 15);
  expect_counts("stencil", stencil, 20, 39);
  expect_counts("periodic", periodic, 20, 45);
  expect_counts("dom", dom, 16, 24);
  expect_counts("tree", tree, 23, 22);
  expect_counts("fft", fft, 32, 58);
  expect_counts("all_to_all", all_to_all, 12, 32);
  expect_counts("nearest", nearest, 20, 39);
  expect_counts("spread", spread, 32, 96);
  expect_counts("random_nearest", random_nearest, 32, 65);

  expect_predecessors("trivial", trivial, 1, 2, {});
  expect_predecessors("no_comm", no_comm, 1, 2, {2});
  expect_predecessors("stencil-left", stencil, 1, 0, {0, 1});
  expect_predecessors("stencil-center", stencil, 2, 2, {1, 2, 3});
  expect_predecessors("periodic-order", periodic, 1, 0, {0, 1, 4});
  expect_predecessors("dom-offset", dom, 4, 1, {0, 1});
  expect_predecessors("tree-parent", tree, 3, 5, {2});
  expect_predecessors("fft-set", fft, 2, 3, {1, 3, 5});
  expect_predecessors(
      "all-to-all-order", all_to_all, 1, 2, {0, 1, 2, 3});
  expect_predecessors("nearest-order", nearest, 1, 2, {1, 2, 3});
  expect_predecessors("spread-order", spread, 1, 1, {1, 4, 6, 0});

  if (stencil.topology_hash != "7936d1f20dfcf2f5") {
    throw std::runtime_error("stencil golden topology hash changed: " +
                             stencil.topology_hash);
  }
  if (trivial.topology_hash == no_comm.topology_hash ||
      stencil.topology_hash == periodic.topology_hash) {
    throw std::runtime_error(
        "different expanded DAGs produced equal topology hashes");
  }

  const ExpandedDag stencil_different_execution =
      build_dag({"-steps", "4", "-width", "5", "-type", "stencil_1d",
                 "-field", "2", "-output", "32",
                 "-kernel", "compute_bound", "-iter", "100"});
  if (stencil.topology_hash !=
      stencil_different_execution.topology_hash) {
    throw std::runtime_error(
        "topology hash changed with execution configuration");
  }
}

void test_data_accesses()
{
  const ExpandedDag no_comm =
      build_dag({"-steps", "3", "-width", "3", "-type", "no_comm",
                 "-field", "1"});
  const DagTask &aliased_task = find_dag_task(no_comm, 1, 2);
  expect_data_access(
      "aliased output", output_data_access(no_comm, aliased_task),
      0, 0, 2, DataAccessMode::write);
  expect_data_access(
      "aliased input",
      input_data_access(no_comm, aliased_task.predecessors[0]),
      0, 0, 2, DataAccessMode::read);

  const ExpandedDag stencil =
      build_dag({"-steps", "3", "-width", "5", "-type", "stencil_1d",
                 "-field", "2"});
  const DagTask &stencil_task = find_dag_task(stencil, 2, 2);
  expect_data_access(
      "stencil output", output_data_access(stencil, stencil_task),
      0, 0, 2, DataAccessMode::write);
  const std::int64_t expected_points[] = {1, 2, 3};
  for (std::size_t i = 0; i < 3; ++i) {
    expect_data_access(
        "ordered stencil input",
        input_data_access(stencil, stencil_task.predecessors[i]),
        0, 1, expected_points[i], DataAccessMode::read);
  }

  const PreparedRun multi = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "trivial", "-and",
       "-steps", "2", "-width", "2", "-type", "no_comm"});
  const DagTask &second_dag_task =
      find_dag_task(multi.expanded_dags[1], 1, 0);
  expect_data_access(
      "second DAG output",
      output_data_access(multi.expanded_dags[1], second_dag_task),
      1, 1, 0, DataAccessMode::write);
}

void expect_parse_failure(const std::vector<std::string> &arguments,
                          const std::string &expected)
{
  try {
    (void)parse_backend_arguments(arguments);
  } catch (const std::runtime_error &error) {
    if (std::string(error.what()).find(expected) != std::string::npos) {
      return;
    }
    throw std::runtime_error(
        "unexpected argument error: " + std::string(error.what()));
  }
  throw std::runtime_error("expected backend argument parsing to fail");
}

void test_arguments()
{
  const Arguments arguments = parse_backend_arguments(
      {"-steps", "3", "-cuda-devices", "3,1",
       "-cuda-placement", "cyclic",
       "-cuda-blocks-per-task", "4",
       "-cuda-compute-dtype", "fp64", "-and",
       "-steps", "2", "-kernel", "busy_wait",
       "-cuda-threads-per-block", "64",
       "-cuda-shmem-bytes-per-block", "1024"});
  if (arguments.gpu_kernel_configs.size() != 2) {
    throw std::runtime_error("expected two GPU kernel configurations");
  }
  if (arguments.run.logical_data_allocator != "cached") {
    throw std::runtime_error(
        "default logical data allocator mismatch");
  }
  if (arguments.run.placement.devices != std::vector<int>({3, 1}) ||
      arguments.run.placement.policy != PlacementPolicy::cyclic) {
    throw std::runtime_error("global task placement configuration mismatch");
  }
  const GpuKernelConfig &first = arguments.gpu_kernel_configs[0];
  const GpuKernelConfig &second = arguments.gpu_kernel_configs[1];
  if (first.launch.blocks_per_task != 4 ||
      first.launch.threads_per_block != 128 ||
      first.launch.dynamic_shared_memory_bytes != 0 ||
      first.compute_data_type != ComputeDataType::fp64) {
    throw std::runtime_error("first GPU kernel configuration mismatch");
  }
  if (second.launch.blocks_per_task != 32 ||
      second.launch.threads_per_block != 64 ||
      second.launch.dynamic_shared_memory_bytes != 1024 ||
      second.compute_data_type != ComputeDataType::fp32) {
    throw std::runtime_error(
        "GPU kernel defaults were not reset after -and");
  }

  for (const std::string &argument : arguments.core_arguments) {
    if (argument.rfind("-cuda-", 0) == 0) {
      throw std::runtime_error(
          "CUDASTF option leaked into Task Bench arguments");
    }
  }

  expect_parse_failure(
      {"-cuda-blocks-per-task", "0"}, "must be greater than zero");
  expect_parse_failure(
      {"-cuda-threads-per-block", "-1"}, "invalid value");
  expect_parse_failure(
      {"-cuda-shmem-bytes-per-block", "-1"}, "invalid value");
  expect_parse_failure(
      {"-cuda-compute-dtype", "fp16"}, "expected fp32 or fp64");
  expect_parse_failure(
      {"-cuda-input-read-policy", "fixed"},
      "unknown CUDASTF option");
  expect_parse_failure(
      {"-cuda-devices", ""}, "requires a non-empty device list");
  expect_parse_failure(
      {"-cuda-devices", "0,"}, "invalid value");
  expect_parse_failure(
      {"-cuda-devices", "0,-1"}, "invalid value");
  expect_parse_failure(
      {"-cuda-devices", "0,0"}, "duplicate device");
  expect_parse_failure(
      {"-cuda-placement", "random"}, "expected block or cyclic");
  expect_parse_failure(
      {"-cuda-device", "0", "-cuda-devices", "0,1"},
      "cannot be used together");
  expect_parse_failure(
      {"-cuda-devices", "0,1", "-cuda-device", "0"},
      "cannot be used together");
}

void test_task_placement()
{
  const ExpandedDag fixed =
      build_dag({"-steps", "3", "-width", "10", "-type", "no_comm"});
  TaskPlacement block{{7, 3, 5}, PlacementPolicy::block};
  const int expected_block[] = {7, 7, 7, 7, 3, 3, 3, 5, 5, 5};
  for (long point = 0; point < 10; ++point) {
    const TaskCoordinates coordinates{0, 0, point};
    if (block.device_for(fixed.task_graph, coordinates) !=
        expected_block[point]) {
      throw std::runtime_error("block task placement mismatch");
    }
  }

  TaskPlacement cyclic{{7, 3, 5}, PlacementPolicy::cyclic};
  for (long point = 0; point < 10; ++point) {
    const TaskCoordinates coordinates{0, 0, point};
    if (cyclic.device_for(fixed.task_graph, coordinates) !=
        cyclic.devices[static_cast<std::size_t>(point) %
                       cyclic.devices.size()]) {
      throw std::runtime_error("cyclic task placement mismatch");
    }
  }

  const ExpandedDag dynamic =
      build_dag({"-steps", "9", "-width", "8", "-type", "dom"});
  for (const DagTask &task : dynamic.tasks) {
    const TaskCoordinates first_timestep{
        task.coordinates.dag_index, 0, task.coordinates.point};
    if (block.device_for(dynamic.task_graph, task.coordinates) !=
        block.device_for(dynamic.task_graph, first_timestep)) {
      throw std::runtime_error(
          "task placement changed for a point across timesteps");
    }
  }

  const std::size_t expected_block_counts[][2] = {
      {1, 0}, {2, 0}, {3, 0}, {2, 2}, {1, 4},
      {0, 4}, {0, 3}, {0, 2}, {0, 1}};
  const std::size_t expected_cyclic_counts[][2] = {
      {1, 0}, {1, 1}, {1, 2}, {2, 2}, {2, 3},
      {2, 2}, {1, 2}, {1, 1}, {0, 1}};
  TaskPlacement dynamic_block{{0, 1}, PlacementPolicy::block};
  TaskPlacement dynamic_cyclic{{0, 1}, PlacementPolicy::cyclic};
  std::size_t block_counts[9][2] = {};
  std::size_t cyclic_counts[9][2] = {};
  for (const DagTask &task : dynamic.tasks) {
    const std::size_t timestep =
        static_cast<std::size_t>(task.coordinates.timestep);
    const int block_device =
        dynamic_block.device_for(dynamic.task_graph, task.coordinates);
    const int cyclic_device =
        dynamic_cyclic.device_for(dynamic.task_graph, task.coordinates);
    ++block_counts[timestep][block_device];
    ++cyclic_counts[timestep][cyclic_device];
  }
  for (std::size_t timestep = 0; timestep < 9; ++timestep) {
    for (std::size_t device = 0; device < 2; ++device) {
      if (block_counts[timestep][device] !=
              expected_block_counts[timestep][device] ||
          cyclic_counts[timestep][device] !=
              expected_cyclic_counts[timestep][device]) {
        throw std::runtime_error(
            "dynamic DAG task distribution mismatch");
      }
    }
  }

  TaskPlacement reordered{{5, 7, 3}, PlacementPolicy::block};
  const TaskCoordinates first{0, 0, 0};
  if (block.device_for(fixed.task_graph, first) ==
      reordered.device_for(fixed.task_graph, first)) {
    throw std::runtime_error("task placement ignored device-list order");
  }
}

std::string execution_config_hash_for(const PreparedRun &run)
{
  return compute_execution_config_hash(
      run.arguments.run, run.arguments.gpu_kernel_configs,
      run.expanded_dags);
}

void test_execution_identity()
{
  const PreparedRun base = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32"});
  const std::string base_hash = execution_config_hash_for(base);
  if (base_hash != "b0da72ba5eec045c") {
    throw std::runtime_error(
        "golden execution config hash changed: " + base_hash);
  }

  const PreparedRun explicit_single = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32", "-cuda-devices", "0"});
  const PreparedRun single_cyclic = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32", "-cuda-device", "0",
       "-cuda-placement", "cyclic"});
  if (base_hash != execution_config_hash_for(explicit_single) ||
      base_hash != execution_config_hash_for(single_cyclic)) {
    throw std::runtime_error(
        "equivalent single-GPU options changed execution config hash");
  }

  const PreparedRun multi_block = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32", "-cuda-devices", "0,1"});
  const PreparedRun multi_reordered = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32", "-cuda-devices", "1,0"});
  const PreparedRun multi_cyclic = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32", "-cuda-devices", "0,1",
       "-cuda-placement", "cyclic"});
  if (base_hash == execution_config_hash_for(multi_block) ||
      execution_config_hash_for(multi_block) ==
          execution_config_hash_for(multi_reordered) ||
      execution_config_hash_for(multi_block) ==
          execution_config_hash_for(multi_cyclic)) {
    throw std::runtime_error(
        "multi-GPU execution config hash ignored placement");
  }
  if (base.expanded_dags[0].topology_hash !=
          multi_block.expanded_dags[0].topology_hash ||
      base.expanded_dags[0].topology_hash !=
          multi_cyclic.expanded_dags[0].topology_hash) {
    throw std::runtime_error(
        "topology hash changed with task placement");
  }

  PreparedRun changed_iterations = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "11",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32"});
  if (base.expanded_dags[0].topology_hash !=
      changed_iterations.expanded_dags[0].topology_hash) {
    throw std::runtime_error(
        "topology hash changed with workload iterations");
  }
  if (base_hash == execution_config_hash_for(changed_iterations)) {
    throw std::runtime_error(
        "execution config hash ignored workload iterations");
  }

  const PreparedRun changed_imbalance = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-imbalance", "1",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32"});
  if (base.expanded_dags[0].topology_hash !=
      changed_imbalance.expanded_dags[0].topology_hash) {
    throw std::runtime_error(
        "topology hash changed with workload imbalance");
  }
  if (base_hash == execution_config_hash_for(changed_imbalance)) {
    throw std::runtime_error(
        "execution config hash ignored workload imbalance");
  }

  std::vector<GpuKernelConfig> changed_config =
      base.arguments.gpu_kernel_configs;
  changed_config[0].launch.blocks_per_task++;
  if (base_hash == compute_execution_config_hash(
          base.arguments.run, changed_config, base.expanded_dags)) {
    throw std::runtime_error(
        "execution config hash ignored kernel launch geometry");
  }
  changed_config = base.arguments.gpu_kernel_configs;
  changed_config[0].compute_data_type = ComputeDataType::fp64;
  if (base_hash == compute_execution_config_hash(
          base.arguments.run, changed_config, base.expanded_dags)) {
    throw std::runtime_error(
        "execution config hash ignored compute data type");
  }
  RunConfig changed_sampling = base.arguments.run;
  changed_sampling.warmup_samples++;
  changed_sampling.measured_samples++;
  changed_sampling.json_path = "different.json";
  if (base_hash != compute_execution_config_hash(
          changed_sampling, base.arguments.gpu_kernel_configs,
          base.expanded_dags)) {
    throw std::runtime_error(
        "execution config hash included sampling or output options");
  }
  RunConfig changed_allocator = base.arguments.run;
  changed_allocator.logical_data_allocator = "uncached";
  if (base_hash == compute_execution_config_hash(
          changed_allocator, base.arguments.gpu_kernel_configs,
          base.expanded_dags)) {
    throw std::runtime_error(
        "execution config hash ignored logical data allocator");
  }

  const PreparedRun empty = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "no_comm"});
  changed_config = empty.arguments.gpu_kernel_configs;
  changed_config[0].compute_data_type = ComputeDataType::fp64;
  if (execution_config_hash_for(empty) != compute_execution_config_hash(
          empty.arguments.run, changed_config, empty.expanded_dags)) {
    throw std::runtime_error(
        "unused compute data type changed execution config hash");
  }

  const PreparedRun graph_order_a = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "trivial", "-and",
       "-steps", "2", "-width", "2", "-type", "no_comm"});
  const PreparedRun graph_order_b = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "no_comm", "-and",
       "-steps", "2", "-width", "2", "-type", "trivial"});
  if (execution_config_hash_for(graph_order_a) ==
      execution_config_hash_for(graph_order_b)) {
    throw std::runtime_error(
        "execution config hash ignored graph order");
  }

  const PreparedRun memory_samples_2 = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "no_comm",
       "-kernel", "memory_bound", "-iter", "4", "-scratch", "64",
       "-sample", "2"});
  const PreparedRun memory_samples_4 = prepare_run(
      {"-steps", "2", "-width", "2", "-type", "no_comm",
       "-kernel", "memory_bound", "-iter", "4", "-scratch", "64",
       "-sample", "4"});
  if (memory_samples_2.expanded_dags[0].topology_hash !=
      memory_samples_4.expanded_dags[0].topology_hash) {
    throw std::runtime_error(
        "topology hash changed with memory sample count");
  }
  if (execution_config_hash_for(memory_samples_2) ==
      execution_config_hash_for(memory_samples_4)) {
    throw std::runtime_error(
        "execution config hash ignored memory sample count");
  }
}

void test_task_iteration_counts()
{
  const ExpandedDag empty =
      build_dag({"-steps", "3", "-width", "2", "-type", "no_comm",
                 "-kernel", "empty", "-iter", "100"});
  const ExpandedDag busy_wait =
      build_dag({"-steps", "3", "-width", "2", "-type", "no_comm",
                 "-kernel", "busy_wait", "-iter", "100"});
  const ExpandedDag memory_bound =
      build_dag({"-steps", "3", "-width", "2", "-type", "no_comm",
                 "-kernel", "memory_bound", "-iter", "2",
                 "-scratch", "64", "-sample", "2"});
  const ExpandedDag compute_bound =
      build_dag({"-steps", "3", "-width", "2", "-type", "no_comm",
                 "-kernel", "compute_bound", "-iter", "100"});
  if (task_iteration_count(empty.task_graph, empty.tasks[0]) != 0 ||
      task_iteration_count(busy_wait.task_graph, busy_wait.tasks[0]) != 100 ||
      task_iteration_count(
          memory_bound.task_graph, memory_bound.tasks[0]) != 2 ||
      task_iteration_count(
          compute_bound.task_graph, compute_bound.tasks[0]) != 100) {
    throw std::runtime_error("GPU workload iteration count mismatch");
  }

  const TaskIterationCounts empty_counts =
      make_task_iteration_counts(empty);
  const TaskIterationCounts compute_counts =
      make_task_iteration_counts(compute_bound);
  const TaskIterationCounts memory_counts =
      make_task_iteration_counts(memory_bound);
  if (empty_counts.size() != empty.tasks.size() ||
      compute_counts.size() != compute_bound.tasks.size() ||
      memory_counts.size() != memory_bound.tasks.size()) {
    throw std::runtime_error("task iteration count size mismatch");
  }
  for (std::uint64_t iterations : empty_counts) {
    if (iterations != 0) {
      throw std::runtime_error("empty task iteration count mismatch");
    }
  }
  for (std::uint64_t iterations : compute_counts) {
    if (iterations != 100) {
      throw std::runtime_error("compute task iteration count mismatch");
    }
  }
  for (std::uint64_t iterations : memory_counts) {
    if (iterations != 2) {
      throw std::runtime_error("memory task iteration count mismatch");
    }
  }

  const ExpandedDag imbalanced =
      build_dag({"-steps", "3", "-width", "3", "-type", "no_comm",
                 "-kernel", "compute_bound", "-iter", "100",
                 "-imbalance", "1"});
  const DagTask &first_imbalanced_task =
      find_dag_task(imbalanced, 0, 0);
  const DagTask &last_imbalanced_task =
      find_dag_task(imbalanced, 2, 2);
  if (task_iteration_count(
          imbalanced.task_graph, first_imbalanced_task) != 144 ||
      task_iteration_count(
          imbalanced.task_graph, last_imbalanced_task) != 61) {
    throw std::runtime_error(
        "deterministic task imbalance golden changed");
  }
  const TaskIterationCounts imbalanced_counts =
      make_task_iteration_counts(imbalanced);
  if (imbalanced_counts.size() != imbalanced.tasks.size() ||
      imbalanced_counts.front() != 144 ||
      imbalanced_counts.back() != 61) {
    throw std::runtime_error(
        "task iteration count order does not match DAG task order");
  }

  const PreparedRun first_launch = prepare_run(
      {"-steps", "3", "-width", "2", "-type", "no_comm",
       "-kernel", "compute_bound", "-iter", "100",
       "-cuda-blocks-per-task", "1",
       "-cuda-threads-per-block", "32"});
  const PreparedRun second_launch = prepare_run(
      {"-steps", "3", "-width", "2", "-type", "no_comm",
       "-kernel", "compute_bound", "-iter", "100",
       "-cuda-blocks-per-task", "64",
       "-cuda-threads-per-block", "256"});
  if (make_task_iteration_counts(first_launch.expanded_dags[0]) !=
      make_task_iteration_counts(second_launch.expanded_dags[0])) {
    throw std::runtime_error(
        "GPU workload iteration count depends on launch geometry");
  }
}

void test_statistics()
{
  SampleResult first;
  first.performance.submission_ms = 4.0;
  first.performance.dag_makespan_ms = 7.0;
  first.diagnostics.setup_ms = 2.0;
  first.diagnostics.sample_total_ms = 10.0;

  SampleResult second;
  second.performance.submission_ms = 2.0;
  second.performance.dag_makespan_ms = 0.045;
  second.diagnostics.setup_ms = 1.0;
  second.diagnostics.sample_total_ms = 6.0;

  const PerformanceSummary summary =
      summarize_performance({first, second});
  expect_near("median submission", summary.median.submission_ms, 3.0);
  expect_near("median DAG makespan", summary.median.dag_makespan_ms, 3.5225);
}

}  // namespace

int main()
{
  try {
    unsetenv("CUDASTF_DEFAULT_ALLOCATOR");
    test_expanded_dag_goldens();
    test_data_accesses();
    test_arguments();
    test_task_placement();
    test_execution_identity();
    test_task_iteration_counts();
    test_statistics();
    std::cout << "CUDASTF host unit tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
