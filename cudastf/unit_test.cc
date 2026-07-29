#include <algorithm>
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
#include "derived_metrics.h"
#include "results.h"
#include "task_profile.h"
#include "topology.h"
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
  std::vector<long> dag_widths;
  dag_widths.reserve(task_bench_app.graphs.size());
  for (const TaskGraph &task_graph : task_bench_app.graphs) {
    dag_widths.push_back(task_graph.max_width);
  }
  parsed.run.stream_pool_size_per_device =
      resolve_stream_pool_size_per_device(
          parsed.run.stream_pool_size_per_device, dag_widths);
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

ExpandedDag make_layered_dag(
    const std::vector<std::uint64_t> &level_widths)
{
  ExpandedDag dag;
  dag.task_graph.graph_index = 0;
  for (std::size_t level = 0; level < level_widths.size(); ++level) {
    for (std::uint64_t point = 0;
         point < level_widths[level]; ++point) {
      DagTask task{
          {0, static_cast<std::int64_t>(level),
           static_cast<std::int64_t>(point)},
          {}};
      if (level != 0) {
        task.predecessors.push_back(
            {static_cast<std::int64_t>(level - 1),
             static_cast<std::int64_t>(
                 point % level_widths[level - 1])});
        ++dag.dependency_edges;
      }
      dag.tasks.push_back(std::move(task));
    }
  }
  return dag;
}

void expect_topology_failure(const char *name, const ExpandedDag &dag,
                             const char *expected)
{
  try {
    (void)compute_topology_metrics({dag});
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(expected) == std::string::npos) {
      throw std::runtime_error(
          std::string(name) + ": unexpected error: " + error.what());
    }
    return;
  }
  throw std::runtime_error(
      std::string(name) + ": expected topology analysis failure");
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
       "-cuda-task-serialization", "enabled",
       "-cuda-task-profiler", "enabled",
       "-cuda-stream-pool-size", "17",
       "-cuda-json", "run.json",
       "-cuda-analysis-json", "analysis.json",
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
  if (arguments.run.task_placement.devices !=
          std::vector<int>({3, 1}) ||
      arguments.run.task_placement.policy !=
          TaskPlacementPolicy::cyclic ||
      arguments.run.stream_pool_size_per_device != 17 ||
      arguments.run.task_serialization != CudaFeatureState::enabled ||
      arguments.run.task_profiler != CudaFeatureState::enabled) {
    throw std::runtime_error("global task placement configuration mismatch");
  }
  if (arguments.output.run_json_path != "run.json" ||
      arguments.output.analysis_json_path != "analysis.json") {
    throw std::runtime_error("output configuration mismatch");
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
      {"-cuda-task-serialization", "yes"},
      "expected disabled or enabled");
  expect_parse_failure(
      {"-cuda-task-profiler", "on"},
      "expected disabled or enabled");
  expect_parse_failure(
      {"-cuda-stream-pool-size", "0"}, "must be greater than zero");
  expect_parse_failure(
      {"-cuda-stream-pool-size", "-1"}, "invalid value");
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
  expect_parse_failure(
      {"-cuda-json", ""}, "requires a non-empty path");
  expect_parse_failure(
      {"-cuda-analysis-json", ""}, "requires a non-empty path");
  expect_parse_failure(
      {"-cuda-json", "same.json",
       "-cuda-analysis-json", "same.json"},
      "require different paths");

  if (resolve_stream_pool_size_per_device(0, {4}) != 4 ||
      resolve_stream_pool_size_per_device(0, {4, 9, 6}) != 9 ||
      resolve_stream_pool_size_per_device(7, {4, 9, 6}) != 7) {
    throw std::runtime_error("stream pool size resolution mismatch");
  }
}

void test_task_placement()
{
  if (std::string(task_placement_policy_name(
          TaskPlacementPolicy::block)) != "block" ||
      std::string(task_placement_policy_description(
          TaskPlacementPolicy::block)) != "contiguous point ranges" ||
      std::string(task_placement_policy_name(
          TaskPlacementPolicy::cyclic)) != "cyclic" ||
      std::string(task_placement_policy_description(
          TaskPlacementPolicy::cyclic)) != "round-robin by point") {
    throw std::runtime_error("task placement policy description mismatch");
  }

  const ExpandedDag fixed =
      build_dag({"-steps", "3", "-width", "10", "-type", "no_comm"});
  TaskPlacement block{{7, 3, 5}, TaskPlacementPolicy::block};
  const int expected_block[] = {7, 7, 7, 7, 3, 3, 3, 5, 5, 5};
  for (long point = 0; point < 10; ++point) {
    const TaskCoordinates coordinates{0, 0, point};
    if (block.device_for(fixed.task_graph, coordinates) !=
        expected_block[point]) {
      throw std::runtime_error("block task placement mismatch");
    }
  }

  TaskPlacement cyclic{{7, 3, 5}, TaskPlacementPolicy::cyclic};
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
  TaskPlacement dynamic_block{{0, 1}, TaskPlacementPolicy::block};
  TaskPlacement dynamic_cyclic{{0, 1}, TaskPlacementPolicy::cyclic};
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

  TaskPlacement reordered{{5, 7, 3}, TaskPlacementPolicy::block};
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

std::string workload_config_hash_for(const PreparedRun &run)
{
  return compute_workload_config_hash(
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
  if (base_hash != "9b8afaac89a4b2f5") {
    throw std::runtime_error(
        "golden execution config hash changed: " + base_hash);
  }
  const std::string base_workload_hash = workload_config_hash_for(base);
  if (base_workload_hash != "dace00b65edd135b") {
    throw std::runtime_error(
        "golden workload config hash changed: " + base_workload_hash);
  }

  const PreparedRun explicit_default_pool = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32",
       "-cuda-stream-pool-size", "5"});
  const PreparedRun changed_pool = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2",
       "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32",
       "-cuda-stream-pool-size", "6"});
  if (execution_config_hash_for(base) !=
          execution_config_hash_for(explicit_default_pool) ||
      workload_config_hash_for(base) !=
          workload_config_hash_for(explicit_default_pool) ||
      execution_config_hash_for(base) ==
          execution_config_hash_for(changed_pool) ||
      workload_config_hash_for(base) ==
          workload_config_hash_for(changed_pool)) {
    throw std::runtime_error(
        "stream pool size execution identity mismatch");
  }

  const PreparedRun multiple_widths = prepare_run(
      {"-steps", "2", "-width", "4", "-type", "trivial", "-and",
       "-steps", "2", "-width", "9", "-type", "no_comm"});
  if (multiple_widths.arguments.run.stream_pool_size_per_device != 9) {
    throw std::runtime_error(
        "multi-DAG stream pool size did not use maximum width");
  }

  const PreparedRun serialized = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2", "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32",
       "-cuda-task-serialization", "enabled"});
  const PreparedRun profiled = prepare_run(
      {"-steps", "4", "-width", "5", "-type", "stencil_1d",
       "-field", "2", "-kernel", "compute_bound", "-iter", "10",
       "-cuda-blocks-per-task", "2", "-cuda-threads-per-block", "64",
       "-cuda-compute-dtype", "fp32",
       "-cuda-task-profiler", "enabled"});
  if (workload_config_hash_for(base) !=
          workload_config_hash_for(serialized) ||
      workload_config_hash_for(base) != workload_config_hash_for(profiled) ||
      base_hash == execution_config_hash_for(serialized) ||
      base_hash == execution_config_hash_for(profiled) ||
      execution_config_hash_for(serialized) ==
          execution_config_hash_for(profiled)) {
    throw std::runtime_error(
        "profiler/serialization hash axes are not orthogonal");
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

void test_topology_metrics()
{
  const ExpandedDag trivial =
      build_dag({"-steps", "4", "-width", "5", "-type", "trivial"});
  const ExpandedDag no_comm =
      build_dag({"-steps", "4", "-width", "5", "-type", "no_comm"});
  const TopologyMetrics fixed =
      compute_topology_metrics({trivial, no_comm});

  const DagTopologyMetrics &trivial_metrics = fixed.dags[0];
  if (trivial_metrics.tasks != 20 ||
      trivial_metrics.dependency_edges != 0 ||
      trivial_metrics.critical_path_length != 1 ||
      trivial_metrics.peak_parallelism != 20) {
    throw std::runtime_error("trivial topology metrics mismatch");
  }
  expect_near(
      "trivial average parallelism",
      trivial_metrics.average_parallelism, 20.0);
  expect_near(
      "trivial p50", trivial_metrics.parallelism_p50, 20.0);
  expect_near(
      "trivial p95", trivial_metrics.parallelism_p95, 20.0);
  expect_near(
      "trivial parallelism CV", trivial_metrics.parallelism_cv, 0.0);

  const DagTopologyMetrics &no_comm_metrics = fixed.dags[1];
  if (no_comm_metrics.tasks != 20 ||
      no_comm_metrics.dependency_edges != 15 ||
      no_comm_metrics.critical_path_length != 4 ||
      no_comm_metrics.peak_parallelism != 5) {
    throw std::runtime_error("no_comm topology metrics mismatch");
  }
  expect_near(
      "no_comm average parallelism",
      no_comm_metrics.average_parallelism, 5.0);
  expect_near(
      "no_comm p50", no_comm_metrics.parallelism_p50, 5.0);
  expect_near(
      "no_comm p95", no_comm_metrics.parallelism_p95, 5.0);
  expect_near(
      "no_comm parallelism CV", no_comm_metrics.parallelism_cv, 0.0);

  const DagTopologyMetrics &fixed_combined = fixed.combined;
  if (fixed_combined.tasks != 40 ||
      fixed_combined.dependency_edges != 15 ||
      fixed_combined.critical_path_length != 4 ||
      fixed_combined.peak_parallelism != 25) {
    throw std::runtime_error("combined fixed topology metrics mismatch");
  }
  expect_near(
      "combined fixed average parallelism",
      fixed_combined.average_parallelism, 10.0);
  expect_near(
      "combined fixed p50", fixed_combined.parallelism_p50, 5.0);
  expect_near(
      "combined fixed p95", fixed_combined.parallelism_p95, 25.0);
  expect_near(
      "combined fixed parallelism CV",
      fixed_combined.parallelism_cv, std::sqrt(0.75));

  const ExpandedDag irregular = make_layered_dag({2, 1, 1});
  const DagTopologyMetrics irregular_metrics =
      compute_topology_metrics({irregular}).dags[0];
  if (irregular_metrics.critical_path_length != 3 ||
      irregular_metrics.peak_parallelism != 2) {
    throw std::runtime_error("irregular topology metrics mismatch");
  }
  expect_near(
      "irregular average parallelism",
      irregular_metrics.average_parallelism, 4.0 / 3.0);
  expect_near(
      "irregular p50", irregular_metrics.parallelism_p50, 1.0);
  expect_near(
      "irregular p95", irregular_metrics.parallelism_p95, 2.0);
  expect_near(
      "irregular parallelism CV",
      irregular_metrics.parallelism_cv, std::sqrt(2.0) / 4.0);

  const ExpandedDag even = make_layered_dag({1, 2, 3, 4});
  const DagTopologyMetrics even_metrics =
      compute_topology_metrics({even}).dags[0];
  expect_near(
      "even-level p50", even_metrics.parallelism_p50, 2.5);
  expect_near(
      "even-level p95", even_metrics.parallelism_p95, 4.0);

  std::vector<std::uint64_t> twenty_level_widths(20);
  for (std::size_t level = 0;
       level < twenty_level_widths.size(); ++level) {
    twenty_level_widths[level] = level + 1;
  }
  const DagTopologyMetrics twenty_level_metrics =
      compute_topology_metrics(
          {make_layered_dag(twenty_level_widths)}).dags[0];
  expect_near(
      "twenty-level p95",
      twenty_level_metrics.parallelism_p95, 19.0);

  const TopologyMetrics combined =
      compute_topology_metrics(
          {irregular, make_layered_dag({1, 3})});
  if (combined.combined.tasks != 8 ||
      combined.combined.dependency_edges != 5 ||
      combined.combined.critical_path_length != 3 ||
      combined.combined.peak_parallelism != 4) {
    throw std::runtime_error("combined irregular topology mismatch");
  }
  expect_near(
      "combined irregular average parallelism",
      combined.combined.average_parallelism, 8.0 / 3.0);
  expect_near(
      "combined irregular p50",
      combined.combined.parallelism_p50, 3.0);
  expect_near(
      "combined irregular p95",
      combined.combined.parallelism_p95, 4.0);

  ExpandedDag duplicate = make_layered_dag({1});
  duplicate.tasks.push_back(duplicate.tasks.front());
  expect_topology_failure(
      "duplicate task", duplicate, "duplicate task");

  ExpandedDag missing;
  missing.task_graph.graph_index = 0;
  missing.tasks.push_back({{0, 1, 0}, {{0, 0}}});
  missing.dependency_edges = 1;
  expect_topology_failure(
      "missing predecessor", missing, "predecessor is missing");

  ExpandedDag edge_mismatch = make_layered_dag({1, 1});
  edge_mismatch.dependency_edges = 0;
  expect_topology_failure(
      "edge mismatch", edge_mismatch, "edge count does not match");
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

SampleResult make_profile_sample(
    const ExpandedDag &dag, const std::vector<double> &durations_ms,
    bool reverse_order)
{
  if (dag.tasks.size() != durations_ms.size()) {
    throw std::logic_error("profile fixture duration count mismatch");
  }
  SampleResult sample;
  sample.task_profile.emplace();
  TaskProfileContext context;
  context.context_id = 17;
  context.task_serialization = CudaFeatureState::enabled;
  context.task_count = dag.tasks.size();
  sample.task_profile->contexts.push_back(context);
  for (std::size_t offset = 0; offset < dag.tasks.size(); ++offset) {
    const std::size_t i = reverse_order
        ? dag.tasks.size() - 1 - offset
        : offset;
    const DagTask &dag_task = dag.tasks[i];
    TaskProfileRecord task;
    task.key = {dag_task.coordinates.dag_index,
                dag_task.coordinates.timestep,
                dag_task.coordinates.point};
    task.configured_device = 0;
    task.context_id = 17;
    task.task_id = static_cast<int>(i + 1);
    task.symbol = "fixture";
    task.has_gpu_activity = true;
    task.start_ns = UINT64_C(10000000) * (i + 1);
    const std::uint64_t duration_ns = static_cast<std::uint64_t>(
        durations_ms[i] * 1.0e6);
    task.end_ns = task.start_ns + duration_ns;
    task.elapsed_ms = durations_ms[i];
    task.operation_count = 1;
    task.device_timings.push_back(
        {0, task.start_ns, task.end_ns, task.elapsed_ms, 1});
    sample.task_profile->tasks.push_back(std::move(task));
  }
  return sample;
}

SampleResult make_runtime_profile_sample(
    const std::vector<ExpandedDag> &dags,
    const std::vector<std::pair<std::uint64_t, std::uint64_t>> &intervals,
    double end_to_end_ms, bool reverse_order)
{
  std::size_t task_count = 0;
  for (const ExpandedDag &dag : dags) task_count += dag.tasks.size();
  if (task_count != intervals.size()) {
    throw std::logic_error("runtime profile fixture interval count mismatch");
  }

  SampleResult sample;
  sample.performance.dag_makespan_ms = end_to_end_ms;
  sample.task_profile.emplace();
  TaskProfileContext context;
  context.context_id = 23;
  context.task_serialization = CudaFeatureState::disabled;
  context.task_count = task_count;
  sample.task_profile->contexts.push_back(context);

  std::size_t index = 0;
  for (const ExpandedDag &dag : dags) {
    for (const DagTask &dag_task : dag.tasks) {
      const auto interval = intervals.at(index);
      TaskProfileRecord task;
      task.key = {dag_task.coordinates.dag_index,
                  dag_task.coordinates.timestep,
                  dag_task.coordinates.point};
      task.configured_device = static_cast<int>(index % 2);
      task.context_id = 23;
      task.task_id = static_cast<int>(index + 1);
      task.symbol = "runtime-fixture";
      task.has_gpu_activity = true;
      task.start_ns = interval.first;
      task.end_ns = interval.second;
      task.elapsed_ms =
          static_cast<double>(interval.second - interval.first) / 1.0e6;
      task.operation_count = 1;
      task.device_timings.push_back(
          {task.configured_device, task.start_ns, task.end_ns,
           task.elapsed_ms, 1});
      sample.task_profile->tasks.push_back(std::move(task));
      ++index;
    }
  }
  if (reverse_order) {
    std::reverse(sample.task_profile->tasks.begin(),
                 sample.task_profile->tasks.end());
  }
  return sample;
}

struct ProfileAssociationFixture {
  TaskActivitySample activity;
  ExpectedProfileTasks expected;
};

ProfileAssociationFixture make_profile_association_fixture()
{
  ProfileAssociationFixture fixture;
  fixture.activity.cupti_timestamp_origin_ns = 42;
  TaskProfileContext context;
  context.context_id = 7;
  context.has_gpu_activity = true;
  context.start_ns = 100;
  context.end_ns = 400;
  context.elapsed_ms = 0.0003;
  context.task_count = 2;
  context.operation_count = 2;
  context.task_serialization = CudaFeatureState::enabled;
  fixture.activity.contexts.push_back(context);

  TaskActivityRecord first;
  first.context_id = 7;
  first.task_id = 11;
  first.symbol = "first";
  first.has_gpu_activity = true;
  first.start_ns = 100;
  first.end_ns = 200;
  first.elapsed_ms = 0.0001;
  first.operation_count = 1;
  first.device_timings.push_back({0, 100, 200, 0.0001, 1});

  TaskActivityRecord second;
  second.context_id = 7;
  second.task_id = 12;
  second.symbol = "second";
  second.has_gpu_activity = true;
  second.start_ns = 300;
  second.end_ns = 400;
  second.elapsed_ms = 0.0001;
  second.operation_count = 1;
  second.device_timings.push_back({1, 300, 400, 0.0001, 1});

  // CUPTI task order is not a logical DAG ordering contract.
  fixture.activity.tasks = {second, first};
  fixture.expected.emplace(
      RuntimeProfileKey{7, 11},
      ExpectedProfileTask{{0, 0, 0}, 0, "first"});
  fixture.expected.emplace(
      RuntimeProfileKey{7, 12},
      ExpectedProfileTask{{0, 0, 1}, 1, "second"});
  return fixture;
}

void expect_profile_failure(
    const char *name, const ProfileAssociationFixture &fixture,
    const char *expected,
    CudaFeatureState serialization = CudaFeatureState::enabled)
{
  try {
    (void)associate_task_profile(
        fixture.activity, fixture.expected, serialization);
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(expected) != std::string::npos) {
      return;
    }
    throw std::runtime_error(
        std::string(name) + ": unexpected error: " + error.what());
  }
  throw std::runtime_error(
      std::string(name) + ": expected profile validation failure");
}

void test_task_profile_association()
{
  const ProfileAssociationFixture valid =
      make_profile_association_fixture();
  const TaskProfileSample associated = associate_task_profile(
      valid.activity, valid.expected, CudaFeatureState::enabled);
  if (associated.tasks.size() != 2 ||
      !(associated.tasks[0].key == DagTaskKey{0, 0, 1}) ||
      !(associated.tasks[1].key == DagTaskKey{0, 0, 0})) {
    throw std::runtime_error(
        "runtime task IDs were not associated with logical tasks");
  }

  ProfileAssociationFixture changed = valid;
  changed.activity.tasks[0].task_id = 99;
  expect_profile_failure("unknown task", changed, "unknown context/task id");

  changed = valid;
  changed.activity.tasks.pop_back();
  expect_profile_failure("missing task", changed, "task count");

  changed = valid;
  changed.activity.tasks[1].task_id = changed.activity.tasks[0].task_id;
  changed.activity.tasks[1].symbol = changed.activity.tasks[0].symbol;
  expect_profile_failure(
      "duplicate task", changed, "duplicate context/task id");

  changed = valid;
  changed.activity.tasks[0].context_id = 8;
  expect_profile_failure("task context", changed, "task context id");

  changed = valid;
  changed.activity.contexts[0].task_count = 3;
  expect_profile_failure("context task count", changed, "task count");

  changed = valid;
  changed.activity.tasks[0].symbol = "wrong";
  expect_profile_failure("task symbol", changed, "symbol mismatch");

  changed = valid;
  changed.activity.tasks[0].device_timings[0].device_id = 0;
  expect_profile_failure("task device", changed, "configured placement");

  changed = valid;
  changed.activity.tasks[0].operation_count = 2;
  expect_profile_failure("task operation count", changed, "invalid GPU activity");

  changed = valid;
  changed.activity.contexts[0].operation_count = 3;
  expect_profile_failure(
      "context operation count", changed, "operation count");

  changed = valid;
  changed.activity.tasks[0].has_gpu_activity = false;
  changed.activity.tasks[0].start_ns = 0;
  changed.activity.tasks[0].end_ns = 0;
  changed.activity.tasks[0].elapsed_ms = 0.0;
  changed.activity.tasks[0].operation_count = 0;
  changed.activity.tasks[0].device_timings.clear();
  changed.activity.contexts[0].operation_count = 1;
  (void)associate_task_profile(
      changed.activity, changed.expected, CudaFeatureState::enabled);
  changed.activity.tasks[0].start_ns = 1;
  expect_profile_failure(
      "inactive task fields", changed, "nonzero timing data");

  changed = valid;
  changed.activity.tasks[0].start_ns = 150;
  changed.activity.tasks[0].device_timings[0].start_ns = 150;
  expect_profile_failure(
      "serialized overlap", changed, "activity overlaps");

  changed.activity.contexts[0].task_serialization =
      CudaFeatureState::disabled;
  (void)associate_task_profile(
      changed.activity, changed.expected, CudaFeatureState::disabled);

  expect_profile_failure(
      "serialization metadata", valid, "serialization metadata mismatch",
      CudaFeatureState::disabled);
}

void test_derived_parallelism_metrics()
{
  const ExpandedDag dag = make_layered_dag({2, 1});
  RunConfig config;
  config.task_profiler = CudaFeatureState::enabled;
  config.task_serialization = CudaFeatureState::enabled;
  const std::vector<SampleResult> samples{
      make_profile_sample(dag, {1.0, 3.0, 2.0}, false),
      make_profile_sample(dag, {3.0, 5.0, 4.0}, true)};

  const ParallelismAnalysis &analysis =
      derive_metrics(config, {dag}, samples).parallelism;
  if (!analysis.available || analysis.dags.size() != 1) {
    throw std::runtime_error("parallelism analysis is unavailable");
  }
  const DagParallelismMetrics &metrics = analysis.dags.front();
  if (metrics.tasks.size() != 3 ||
      metrics.tasks[0].measured_duration_ms != 2.0 ||
      metrics.tasks[1].measured_duration_ms != 4.0 ||
      metrics.tasks[2].measured_duration_ms != 3.0) {
    throw std::runtime_error("measured task median mismatch");
  }
  expect_near(
      "measured work", metrics.parallelism.task_work_ms, 9.0);
  expect_near(
      "weighted critical path", metrics.parallelism.critical_path_ms, 5.0);
  expect_near("DAG parallelism average", metrics.parallelism.average, 1.8);
  if (metrics.parallelism.peak != 2) {
    throw std::runtime_error("DAG parallelism peak mismatch");
  }
  expect_near("DAG parallelism p50", metrics.parallelism.p50, 2.0);
  expect_near("DAG parallelism p95", metrics.parallelism.p95, 2.0);
  expect_near("DAG parallelism CV", metrics.parallelism.cv, 2.0 / 9.0);
  expect_near("duration mean", metrics.task_duration.mean_ms, 3.0);
  expect_near("duration median", metrics.task_duration.median_ms, 3.0);
  expect_near("duration p95", metrics.task_duration.p95_ms, 4.0);
  expect_near(
      "duration CV", metrics.task_duration.cv,
      std::sqrt(2.0 / 3.0) / 3.0);

  ExpandedDag chain = make_layered_dag({1, 1});
  ExpandedDag independent = make_layered_dag({2});
  independent.task_graph.graph_index = 1;
  for (DagTask &task : independent.tasks) {
    task.coordinates.dag_index = 1;
  }
  SampleResult combined_sample;
  combined_sample.task_profile.emplace();
  TaskProfileContext combined_context;
  combined_context.context_id = 19;
  combined_context.task_serialization = CudaFeatureState::enabled;
  combined_context.task_count = 4;
  combined_sample.task_profile->contexts.push_back(combined_context);
  const auto append_tasks =
      [&](const ExpandedDag &fixture_dag,
          const std::vector<double> &durations) {
        for (std::size_t i = 0; i < fixture_dag.tasks.size(); ++i) {
          const DagTask &dag_task = fixture_dag.tasks[i];
          TaskProfileRecord task;
          task.key = {dag_task.coordinates.dag_index,
                      dag_task.coordinates.timestep,
                      dag_task.coordinates.point};
          task.configured_device = 0;
          task.context_id = 19;
          task.task_id = static_cast<int>(
              combined_sample.task_profile->tasks.size() + 1);
          task.symbol = "combined-fixture";
          task.has_gpu_activity = true;
          task.start_ns = UINT64_C(10000000) * task.task_id;
          task.end_ns = task.start_ns + static_cast<std::uint64_t>(
              durations[i] * 1.0e6);
          task.elapsed_ms = durations[i];
          task.operation_count = 1;
          task.device_timings.push_back(
              {0, task.start_ns, task.end_ns, task.elapsed_ms, 1});
          combined_sample.task_profile->tasks.push_back(std::move(task));
        }
      };
  append_tasks(chain, {2.0, 3.0});
  append_tasks(independent, {4.0, 1.0});
  const ParallelismMetrics &combined =
      derive_metrics(config, {chain, independent}, {combined_sample})
          .parallelism.combined.parallelism;
  expect_near(
      "combined measured work", combined.task_work_ms, 10.0);
  expect_near("combined weighted critical path",
              combined.critical_path_ms, 5.0);
  expect_near("combined DAG parallelism average", combined.average, 2.0);
  if (combined.peak != 3) {
    throw std::runtime_error("combined DAG parallelism peak mismatch");
  }
  expect_near("combined DAG parallelism p50", combined.p50, 2.0);
  expect_near("combined DAG parallelism p95", combined.p95, 3.0);

  RunConfig normal = config;
  normal.task_serialization = CudaFeatureState::disabled;
  if (derive_metrics(normal, {dag}, {}).parallelism.available) {
    throw std::runtime_error(
        "normal profiled execution produced DAG parallelism metrics");
  }

  KernelResourcesByDag kernel_resources{{{0, 32, 0, 4}}};
  const std::string first_hash = compute_raw_data_hash(
      "w", "e", "environment", kernel_resources, samples);
  std::vector<SampleResult> changed = samples;
  changed[0].task_profile->tasks[0].end_ns++;
  if (first_hash ==
      compute_raw_data_hash(
          "w", "e", "environment", kernel_resources, changed)) {
    throw std::runtime_error("raw data hash ignored task timing");
  }
  kernel_resources[0][0].max_active_blocks_per_sm++;
  if (first_hash == compute_raw_data_hash(
          "w", "e", "environment", kernel_resources, samples)) {
    throw std::runtime_error("raw data hash ignored kernel resources");
  }
}

void test_kernel_capacity_metrics()
{
  ExpandedDag dag = make_layered_dag({2});
  dag.task_graph.max_width = 2;
  RunConfig config;
  config.task_placement.devices = {0, 1};
  config.task_placement.policy = TaskPlacementPolicy::cyclic;
  GpuKernelConfig kernel;
  kernel.launch.blocks_per_task = 32;
  CudaDeviceInfo first;
  first.device_id = 0;
  first.sm_count = 10;
  CudaDeviceInfo second;
  second.device_id = 1;
  second.sm_count = 12;
  first.name = second.name = "fixture-gpu";
  first.compute_capability_major = second.compute_capability_major = 10;
  first.compute_capability_minor = second.compute_capability_minor = 0;
  const std::string environment_hash =
      compute_environment_hash({first, second});
  second.sm_count++;
  if (environment_hash == compute_environment_hash({first, second})) {
    throw std::runtime_error("environment hash ignored SM count");
  }
  second.sm_count--;
  const KernelResourcesByDag resources{
      {{0, 32, 0, 4}, {1, 32, 0, 5}}};
  const KernelCapacityMetrics capacity = compute_kernel_capacity(
      config, {dag}, {kernel}, {first, second}, resources);
  const DagKernelCapacity &result = capacity.dags.front();
  if (result.devices.size() != 2 ||
      result.devices[0].resident_blocks != 40 ||
      result.devices[0].occupancy_saturation_tasks != 2 ||
      result.devices[1].resident_blocks != 60 ||
      result.devices[1].occupancy_saturation_tasks != 2 ||
      result.combined.device_count != 2 ||
      result.combined.resident_blocks != 100 ||
      result.combined.occupancy_saturation_tasks != 4) {
    throw std::runtime_error("multi-GPU kernel capacity mismatch");
  }

  dag.task_graph.max_width = 1;
  dag.tasks.resize(1);
  const KernelCapacityMetrics single_capacity = compute_kernel_capacity(
      config, {dag}, {kernel}, {first, second}, resources);
  const DagKernelCapacity &single_device = single_capacity.dags.front();
  if (single_device.devices.size() != 1 ||
      single_device.devices.front().device_id != 0 ||
      single_device.combined.device_count != 1 ||
      single_device.combined.occupancy_saturation_tasks != 2) {
    throw std::runtime_error(
        "kernel capacity included an unassigned device");
  }
}

void test_derived_concurrency_metrics()
{
  const ExpandedDag dag = make_layered_dag({2});
  RunConfig config;
  config.task_profiler = CudaFeatureState::enabled;
  config.task_serialization = CudaFeatureState::disabled;
  const SampleResult gapped = make_runtime_profile_sample(
      {dag}, {{UINT64_C(1000000), UINT64_C(3000000)},
              {UINT64_C(6000000), UINT64_C(8000000)}},
      10.0, false);
  const SampleResult overlapping = make_runtime_profile_sample(
      {dag}, {{UINT64_C(1000000001), UINT64_C(1004000001)},
              {UINT64_C(1001000001), UINT64_C(1005000001)}},
      8.0, true);

  const ConcurrencyAnalysis &analysis =
      derive_metrics(config, {dag}, {gapped, overlapping}).concurrency;
  if (!analysis.available || analysis.samples.size() != 2 ||
      analysis.summary.sample_count != 2) {
    throw std::runtime_error("concurrency analysis is unavailable");
  }
  const CombinedConcurrencyMetrics &first =
      analysis.samples[0].combined;
  expect_near("runtime task work", first.task_work_ms, 4.0);
  expect_near("Task GPU span", first.task_gpu_span.duration_ms, 7.0);
  expect_near("Task GPU average", first.task_gpu_span.average, 4.0 / 7.0);
  if (first.task_gpu_span.peak != 1) {
    throw std::runtime_error("Task GPU peak mismatch");
  }
  expect_near("Task GPU p50", first.task_gpu_span.p50, 1.0);
  expect_near("Task GPU p95", first.task_gpu_span.p95, 1.0);
  expect_near("Task GPU CV", first.task_gpu_span.cv, std::sqrt(3.0) / 2.0);
  if (!first.end_to_end_span.available) {
    throw std::runtime_error("End-to-end span is unavailable");
  }
  const ConcurrencyMetrics &end_to_end =
      first.end_to_end_span.metrics;
  expect_near("End-to-end span", end_to_end.duration_ms, 10.0);
  expect_near("End-to-end average", end_to_end.average, 0.4);
  expect_near("End-to-end p50", end_to_end.p50, 0.0);
  expect_near("End-to-end p95", end_to_end.p95, 1.0);
  expect_near("End-to-end CV", end_to_end.cv, std::sqrt(1.5));

  const CombinedConcurrencySummary &summary = analysis.summary.combined;
  expect_near("summary task work median", summary.task_work_ms.median, 6.0);
  expect_near("summary task work p95", summary.task_work_ms.p95, 8.0);
  expect_near(
      "summary Task GPU span median",
      summary.task_gpu_span.duration_ms.median, 6.0);
  expect_near(
      "summary Task GPU peak median",
      summary.task_gpu_span.peak.median, 1.5);
  if (!summary.end_to_end_span.available) {
    throw std::runtime_error("End-to-end summary is unavailable");
  }
  expect_near(
      "summary End-to-end span median",
      summary.end_to_end_span.metrics.duration_ms.median, 9.0);

  SampleResult invalid_end_to_end = gapped;
  invalid_end_to_end.performance.dag_makespan_ms = 6.0;
  const ConcurrencyAnalysis &partial =
      derive_metrics(config, {dag}, {invalid_end_to_end}).concurrency;
  if (!partial.available ||
      partial.samples[0].combined.end_to_end_span.available ||
      partial.summary.combined.end_to_end_span.available) {
    throw std::runtime_error(
        "invalid End-to-end span discarded valid Task GPU metrics");
  }
  expect_near(
      "partial Task GPU span",
      partial.samples[0].combined.task_gpu_span.duration_ms, 7.0);

  SampleResult inactive = gapped;
  TaskProfileRecord &inactive_task = inactive.task_profile->tasks[0];
  inactive_task.has_gpu_activity = false;
  inactive_task.start_ns = 0;
  inactive_task.end_ns = 0;
  inactive_task.elapsed_ms = 0.0;
  inactive_task.operation_count = 0;
  inactive_task.device_timings.clear();
  const ConcurrencyAnalysis &inactive_analysis =
      derive_metrics(config, {dag}, {inactive}).concurrency;
  if (inactive_analysis.available ||
      inactive_analysis.unavailable_reason.find("no correlated GPU activity") ==
          std::string::npos) {
    throw std::runtime_error(
        "inactive runtime task produced concurrency metrics");
  }

  const SampleResult adjacent = make_runtime_profile_sample(
      {dag}, {{UINT64_C(1000), UINT64_C(2001000)},
              {UINT64_C(2001000), UINT64_C(4001000)}},
      5.0, false);
  const ConcurrencyMetrics &adjacent_span =
      derive_metrics(config, {dag}, {adjacent})
          .concurrency.samples[0].combined.task_gpu_span;
  expect_near("adjacent Task GPU average", adjacent_span.average, 1.0);
  if (adjacent_span.peak != 1) {
    throw std::runtime_error("half-open task intervals overlap at boundary");
  }

  ExpandedDag second_dag = make_layered_dag({1});
  second_dag.task_graph.graph_index = 1;
  for (DagTask &task : second_dag.tasks) {
    task.coordinates.dag_index = 1;
  }
  ExpandedDag first_dag = make_layered_dag({1});
  const SampleResult cross_dag = make_runtime_profile_sample(
      {first_dag, second_dag},
      {{UINT64_C(1000), UINT64_C(10001000)},
       {UINT64_C(1000), UINT64_C(10001000)}},
      11.0, false);
  const ConcurrencyAnalysis &cross_dag_analysis =
      derive_metrics(
          config, {first_dag, second_dag}, {cross_dag}).concurrency;
  if (cross_dag_analysis.samples[0].dags[0].task_gpu_span.peak != 1 ||
      cross_dag_analysis.samples[0].dags[1].task_gpu_span.peak != 1 ||
      cross_dag_analysis.samples[0].combined.task_gpu_span.peak != 2) {
    throw std::runtime_error("combined DAG concurrency lost real overlap");
  }

  RunConfig serialized = config;
  serialized.task_serialization = CudaFeatureState::enabled;
  if (derive_metrics(serialized, {dag}, {}).concurrency.available) {
    throw std::runtime_error(
        "serialized execution produced concurrency metrics");
  }
  RunConfig unprofiled = config;
  unprofiled.task_profiler = CudaFeatureState::disabled;
  if (derive_metrics(unprofiled, {dag}, {gapped}).concurrency.available) {
    throw std::runtime_error(
        "unprofiled execution produced concurrency metrics");
  }
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
    test_kernel_capacity_metrics();
    test_topology_metrics();
    test_statistics();
    test_task_profile_association();
    test_derived_parallelism_metrics();
    test_derived_concurrency_metrics();
    std::cout << "CUDASTF host unit tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
