#!/usr/bin/env bash

set -euo pipefail
ulimit -c 0
unset CUDASTF_DEFAULT_ALLOCATOR CUDASTF_CACHED_FIFO \
  USER_ALLOC_POOLS_MEM_CAP

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
task_bench="$script_dir/task_bench"
test_tmp=$(mktemp -d)
trap 'rm -rf "$test_tmp"' EXIT

run_case()
{
  local name=$1
  shift
  "$task_bench" "$@" -cuda-warmup 0 -cuda-runs 1 \
    >"$test_tmp/$name.out" 2>&1
}

expect_failure()
{
  local name=$1
  local expected=$2
  shift 2
  if "$task_bench" "$@" -cuda-warmup 0 -cuda-runs 1 \
      >"$test_tmp/$name.out" 2>&1; then
    echo "expected failure but command succeeded: $name" >&2
    return 1
  fi
  if ! grep -F "$expected" "$test_tmp/$name.out" >/dev/null; then
    echo "failure output did not contain '$expected': $name" >&2
    sed -n '1,80p' "$test_tmp/$name.out" >&2
    return 1
  fi
}

run_case trivial -steps 5 -width 8 -type trivial
run_case no_comm_field1 \
  -steps 5 -width 8 -type no_comm -field 1 -output 32
run_case stencil -steps 5 -width 8 -type stencil_1d -field 2
run_case periodic \
  -steps 5 -width 8 -type stencil_1d_periodic -field 2
run_case dom -steps 9 -width 5 -type dom
run_case tree -steps 6 -width 8 -type tree
run_case fft -steps 6 -width 8 -type fft
run_case all_to_all_9 -steps 3 -width 9 -type all_to_all
run_case nearest -steps 5 -width 9 -type nearest -radix 9
run_case spread \
  -steps 6 -width 8 -type spread -radix 4 -period 2
run_case random_nearest \
  -steps 6 -width 8 -type random_nearest -radix 8 -period 3 \
  -fraction 0.5
for fanin in {1..9}; do
  run_case "fanin_$fanin" \
    -steps 2 -width "$fanin" -type all_to_all
done
run_case fanin_32 \
  -steps 2 -width 32 -type all_to_all
run_case odd_output_bytes \
  -steps 3 -width 2 -type no_comm -output 17
run_case multi_graph \
  -steps 5 -width 8 -type fft -and \
  -steps 6 -width 8 -type tree
run_case busy_wait \
  -steps 3 -width 4 -type no_comm -field 1 \
  -kernel busy_wait -iter 4096 \
  -cuda-blocks-per-task 2 -cuda-threads-per-block 64 \
  -cuda-shmem-bytes-per-block 1024
run_case compute_fp32 \
  -steps 3 -width 4 -type no_comm -field 1 \
  -kernel compute_bound -iter 4096 \
  -cuda-blocks-per-task 2 -cuda-threads-per-block 64 \
  -cuda-compute-dtype fp32
run_case compute_fp64 \
  -steps 3 -width 4 -type no_comm -field 1 \
  -kernel compute_bound -iter 4096 \
  -cuda-blocks-per-task 2 -cuda-threads-per-block 64 \
  -cuda-compute-dtype fp64
run_case memory_bound \
  -steps 3 -width 4 -type no_comm -field 1 \
  -kernel memory_bound -iter 4 -output 64 \
  -scratch 65536 -sample 4 \
  -cuda-blocks-per-task 2 -cuda-threads-per-block 64
run_case temporary_scratch_lifetime \
  -steps 1024 -width 1 -type no_comm -field 1 \
  -kernel memory_bound -iter 0 -scratch 134217728 -sample 1

"$task_bench" -h >"$test_tmp/help.out" 2>&1
grep -F "every predecessor input is read completely once" \
  "$test_tmp/help.out" >/dev/null

run_case verbose -v -steps 3 -width 100 -type no_comm
grep -F "Configuration:" "$test_tmp/verbose.out" >/dev/null
if grep -F "Timestep " "$test_tmp/verbose.out" >/dev/null; then
  echo "-v unexpectedly printed timestep details" >&2
  exit 1
fi

run_case extra_verbose -vv -steps 3 -width 100 -type no_comm
grep -F "Timestep 0: points 0..99 (100), previous none (0)" \
  "$test_tmp/extra_verbose.out" >/dev/null
grep -F "100 dependency edges; fan-in 1; examples:" \
  "$test_tmp/extra_verbose.out" >/dev/null
if test "$(wc -l <"$test_tmp/extra_verbose.out")" -ge 80; then
  echo "-vv output is unexpectedly verbose" >&2
  exit 1
fi

run_case full_verbose -vvv -steps 2 -width 3 -type no_comm
grep -F "Dependencies:" "$test_tmp/full_verbose.out" >/dev/null
grep -F "Reverse Dependencies:" "$test_tmp/full_verbose.out" >/dev/null

"$task_bench" -steps 5 -width 8 -type stencil_1d -field 2 \
  -cuda-warmup 0 -cuda-runs 2 -cuda-json "$test_tmp/result-a.json" \
  >"$test_tmp/json-a.out" 2>&1
"$task_bench" -steps 5 -width 8 -type stencil_1d -field 2 \
  -cuda-warmup 0 -cuda-runs 1 -cuda-json "$test_tmp/result-b.json" \
  >"$test_tmp/json-b.out" 2>&1
CUDASTF_DEFAULT_ALLOCATOR=uncached \
  "$task_bench" -steps 5 -width 8 -type stencil_1d -field 2 \
    -cuda-warmup 0 -cuda-runs 1 \
    -cuda-json "$test_tmp/result-uncached.json" \
    >"$test_tmp/json-uncached.out" 2>&1
"$task_bench" -steps 5 -width 8 -type stencil_1d -field 2 \
  -kernel compute_bound -iter 4096 \
  -cuda-blocks-per-task 2 -cuda-threads-per-block 64 \
  -cuda-compute-dtype fp32 \
  -cuda-warmup 0 -cuda-runs 1 \
  -cuda-json "$test_tmp/result-workload.json" \
  >"$test_tmp/json-workload.out" 2>&1
"$task_bench" \
  -steps 3 -width 4 -type no_comm -field 1 \
  -kernel memory_bound -iter 4 -scratch 65536 -sample 4 \
  -cuda-warmup 0 -cuda-runs 1 \
  -cuda-json "$test_tmp/result-memory.json" \
  >"$test_tmp/json-memory.out" 2>&1
"$task_bench" \
  -steps 2 -width 2 -type no_comm -field 1 \
  -kernel compute_bound -iter 128 \
  -cuda-blocks-per-task 3 -cuda-threads-per-block 64 \
  -cuda-compute-dtype fp64 -and \
  -steps 2 -width 2 -type no_comm -field 1 \
  -kernel busy_wait -iter 64 \
  -cuda-blocks-per-task 5 -cuda-threads-per-block 32 \
  -cuda-shmem-bytes-per-block 256 \
  -cuda-warmup 0 -cuda-runs 1 \
  -cuda-json "$test_tmp/result-multi.json" \
  >"$test_tmp/json-multi.out" 2>&1

default_shmem=$(
  jq -r '.device.legacy_shared_memory_per_block_bytes' \
    "$test_tmp/result-a.json"
)
optin_shmem=$(
  jq -r '.device.optin_shared_memory_per_block_bytes' \
    "$test_tmp/result-a.json"
)
if ((optin_shmem > default_shmem)); then
  requested_shmem=$((default_shmem + 1024))
  if ((requested_shmem > optin_shmem)); then
    requested_shmem=$optin_shmem
  fi
  run_case optin_shared_memory \
    -steps 2 -width 2 -type no_comm \
    -cuda-shmem-bytes-per-block "$requested_shmem"
  run_case optin_shared_memory_multi_dag \
    -steps 2 -width 2 -type no_comm \
    -cuda-shmem-bytes-per-block "$requested_shmem" -and \
    -steps 2 -width 2 -type no_comm
fi

jq -e '
  .format == "cudastf-task-bench-results" and
  .backend == "cudastf" and
  (.execution_config_hash | test("^[0-9a-f]{16}$")) and
  .run_config.context == "stream" and
  .run_config.logical_data_allocator == "cached" and
  .run_config.device_id == 0 and
  .run_config.warmup_samples == 0 and
  .run_config.measured_samples == 2 and
  (.task_bench_revision | length) > 0 and
  (.task_bench_worktree_dirty | type) == "boolean" and
  (.cccl_revision | length) > 0 and
  (.cccl_worktree_dirty | type) == "boolean" and
  (.build.type == "release" or .build.type == "debug") and
  (.build.cuda_arch_option | length) > 0 and
  (.build.cuda_arch_resolved | length) > 0 and
  .device.device_id == 0 and
  (.device.compute_capability | test("^[0-9]+\\.[0-9]+$")) and
  .device.legacy_shared_memory_per_block_bytes > 0 and
  .device.optin_shared_memory_per_block_bytes >=
    .device.legacy_shared_memory_per_block_bytes and
  (.dags | length) == 1 and
  .dags[0].dag_index == 0 and
  .dags[0].timesteps == 5 and
  .dags[0].max_width == 8 and
  .dags[0].dependence == "stencil_1d" and
  .dags[0].radix == 3 and
  .dags[0].period == 0 and
  .dags[0].fraction_connected == 0.25 and
  .dags[0].nb_fields == 2 and
  .dags[0].output_bytes_per_task == 16 and
  .dags[0].scratch_bytes_per_task == 0 and
  .dags[0].kernel.type == "empty" and
  .dags[0].kernel.compute_data_type == null and
  .dags[0].kernel.launch.blocks_per_task == 32 and
  .dags[0].kernel.launch.threads_per_block == 128 and
  .dags[0].kernel.launch.dynamic_shared_memory_bytes == 0 and
  .dags[0].kernel.work_model.logical_iterations == 0 and
  .dags[0].kernel.work_model.task_input_read_bytes == 1408 and
  .dags[0].kernel.work_model.task_output_write_bytes == 640 and
  .dags[0].kernel.work_model.scratch_read_bytes == 0 and
  .dags[0].kernel.work_model.scratch_write_bytes == 0 and
  (.dags[0].kernel.work_model | has("scratch_store_bytes") | not) and
  (.dags[0].topology_hash | test("^[0-9a-f]{16}$")) and
  .dags[0].tasks == 40 and
  .dags[0].dependency_edges == 88 and
  .dags[0].task_data_accesses == 128 and
  (.dags[0] | has("scratch_accesses") | not) and
  .dags[0].max_fanin == 3 and
  .dags[0].task_data_store_bytes == 256 and
  (.dags[0] | has("scratch_store_bytes") | not) and
  .aggregate_counts.tasks == 40 and
  .aggregate_counts.dependency_edges == 88 and
  .aggregate_counts.task_data_accesses == 128 and
  .aggregate_counts.task_data_store_bytes == 256 and
  (.aggregate_counts | has("scratch_accesses") | not) and
  (.aggregate_counts | has("scratch_store_bytes") | not) and
  (.samples | length) == 2 and
  (.samples[0].submission_ms | type) == "number" and
  (.samples[0].dag_makespan_ms | type) == "number" and
  (.samples[0].diagnostics.setup_ms | type) == "number" and
  (.samples[0].diagnostics.sample_total_ms | type) == "number" and
  (.summary.median | keys | sort) ==
    ["dag_makespan_ms", "submission_ms"] and
  .summary.min.dag_makespan_ms <= .summary.p95.dag_makespan_ms
' "$test_tmp/result-a.json" >/dev/null

topology_a=$(jq -r '.dags[0].topology_hash' "$test_tmp/result-a.json")
topology_b=$(jq -r '.dags[0].topology_hash' "$test_tmp/result-b.json")
topology_workload=$(
  jq -r '.dags[0].topology_hash' "$test_tmp/result-workload.json"
)
test "$topology_a" = "$topology_b"
test "$topology_a" = "$topology_workload"

execution_a=$(jq -r '.execution_config_hash' "$test_tmp/result-a.json")
execution_b=$(jq -r '.execution_config_hash' "$test_tmp/result-b.json")
execution_workload=$(
  jq -r '.execution_config_hash' "$test_tmp/result-workload.json"
)
execution_uncached=$(
  jq -r '.execution_config_hash' "$test_tmp/result-uncached.json"
)
test "$execution_a" = "$execution_b"
test "$execution_a" != "$execution_workload"
test "$execution_a" != "$execution_uncached"
jq -e '.run_config.logical_data_allocator == "uncached"' \
  "$test_tmp/result-uncached.json" >/dev/null
grep -F "Logical data allocator: uncached" \
  "$test_tmp/json-uncached.out" >/dev/null

jq -e '
  .dags[0].kernel.type == "compute_bound" and
  .dags[0].kernel.compute_data_type == "fp32" and
  .dags[0].kernel.work_model.logical_iterations == 163840 and
  .dags[0].kernel.work_model.task_input_read_bytes == 1408 and
  .dags[0].kernel.work_model.task_output_write_bytes == 640 and
  .dags[0].kernel.work_model.scratch_read_bytes == 0 and
  .dags[0].kernel.work_model.scratch_write_bytes == 0
' "$test_tmp/result-workload.json" >/dev/null

jq -e '
  .dags[0].kernel.type == "memory_bound" and
  .dags[0].kernel.compute_data_type == null and
  .dags[0].kernel.samples == 4 and
  .dags[0].kernel.work_model.logical_iterations == 48 and
  .dags[0].kernel.work_model.task_input_read_bytes == 128 and
  .dags[0].kernel.work_model.task_output_write_bytes == 192 and
  .dags[0].kernel.work_model.scratch_read_bytes == 393216 and
  .dags[0].kernel.work_model.scratch_write_bytes == 393216 and
  (.dags[0].kernel.work_model | has("scratch_store_bytes") | not) and
  (.dags[0] | has("scratch_accesses") | not) and
  .dags[0].task_data_store_bytes == 64 and
  (.dags[0] | has("scratch_store_bytes") | not)
' "$test_tmp/result-memory.json" >/dev/null

jq -e '
  (.dags | length) == 2 and
  .dags[0].kernel.type == "compute_bound" and
  .dags[0].kernel.compute_data_type == "fp64" and
  .dags[0].kernel.launch.blocks_per_task == 3 and
  .dags[0].kernel.launch.threads_per_block == 64 and
  .dags[0].kernel.launch.dynamic_shared_memory_bytes == 0 and
  .dags[1].kernel.type == "busy_wait" and
  .dags[1].kernel.compute_data_type == null and
  .dags[1].kernel.launch.blocks_per_task == 5 and
  .dags[1].kernel.launch.threads_per_block == 32 and
  .dags[1].kernel.launch.dynamic_shared_memory_bytes == 256
' "$test_tmp/result-multi.json" >/dev/null

expect_failure field1_cross_point "unsupported field reuse" \
  -steps 5 -width 8 -type stencil_1d -field 1
expect_failure all_to_all_33 "supports at most 32 predecessors" \
  -steps 3 -width 33 -type all_to_all -output 33
expect_failure unsupported_kernel \
  "supports '-kernel empty', 'busy_wait', 'memory_bound', and 'compute_bound'" \
  -kernel compute_bound2
expect_failure scratch_without_memory_bound \
  "uses '-scratch' only with '-kernel memory_bound'" \
  -scratch 8
expect_failure memory_bound_without_scratch \
  "memory_bound requires '-scratch' greater than zero" \
  -kernel memory_bound
expect_failure memory_bound_zero_samples \
  "memory_bound requires '-sample' greater than zero" \
  -kernel memory_bound -scratch 64 -sample 0
expect_failure memory_bound_uneven_samples \
  "requires '-scratch' to be divisible by '-sample'" \
  -kernel memory_bound -scratch 65 -sample 4
expect_failure memory_bound_odd_sample \
  "equal source and destination regions" \
  -kernel memory_bound -scratch 12 -sample 4
expect_failure negative_device "invalid value for -cuda-device" \
  -cuda-device -1
expect_failure zero_blocks "must be greater than zero" \
  -cuda-blocks-per-task 0
expect_failure negative_threads "invalid value" \
  -cuda-threads-per-block -1
expect_failure invalid_compute_dtype "expected fp32 or fp64" \
  -cuda-compute-dtype fp16
expect_failure removed_input_read_policy \
  "unknown CUDASTF option" \
  -cuda-input-read-policy fixed
expect_failure too_many_threads "exceeds the selected device limit" \
  -cuda-threads-per-block 100000
expect_failure too_much_shmem "exceeds the selected device limit" \
  -cuda-shmem-bytes-per-block 999999999
expect_failure invalid_device "invalid CUDA device" -cuda-device 9999
max_device=$(
  sed -n 's/.*available range is \[0,\([0-9][0-9]*\)\].*/\1/p' \
    "$test_tmp/invalid_device.out"
)
if [[ -n $max_device ]] && ((max_device >= 1)); then
  run_case device_1 \
    -steps 2 -width 2 -type no_comm -cuda-device 1
  grep -F "Device: 1 (" "$test_tmp/device_1.out" >/dev/null
fi
expect_failure invalid_context "only the stream executor is implemented" \
  -cuda-context graph
expect_failure unknown_cuda_option "unknown CUDASTF option" \
  -cuda-unknown value
if CUDASTF_DEFAULT_ALLOCATOR=invalid \
    "$task_bench" -steps 2 -width 2 -type no_comm \
      -cuda-warmup 0 -cuda-runs 1 \
      >"$test_tmp/invalid-allocator.out" 2>&1; then
  echo "expected invalid allocator selection to fail" >&2
  exit 1
fi
grep -F "invalid CUDASTF_DEFAULT_ALLOCATOR" \
  "$test_tmp/invalid-allocator.out" >/dev/null
expect_failure random_spread "is not implemented" \
  -type random_spread
expect_failure fft_width_1 "requires a width of at least 2" \
  -type fft -width 1
expect_failure periodic_width_1 "requires a width of at least 3" \
  -type stencil_1d_periodic -width 1
expect_failure periodic_width_2 "requires a width of at least 3" \
  -type stencil_1d_periodic -width 2
expect_failure spread_radix_0 "requires radix in [1" \
  -type spread -width 8 -radix 0 -period 1
expect_failure spread_radix_too_large "requires radix in [1" \
  -type spread -width 8 -radix 9 -period 1

if command -v dot >/dev/null; then
  "$script_dir/render_dag.sh" "$test_tmp/dag.png" -- \
    -steps 2 -width 2 -type no_comm \
    >"$test_tmp/render-dag.out" 2>&1
  test -s "$test_tmp/dag.dot"
  test -s "$test_tmp/dag.png"
  grep -F "digraph {" "$test_tmp/dag.dot" >/dev/null
  grep -F "T(empty,(0,0,0))" \
    "$test_tmp/dag.dot" >/dev/null
  grep -F "D(0,0,0)(write)(16)" \
    "$test_tmp/dag.dot" >/dev/null

  CUDASTF_DOT_FILE="$test_tmp/memory.dot" \
    "$task_bench" -steps 2 -width 2 -type no_comm -field 1 \
      -kernel memory_bound -iter 2 -scratch 64 -sample 2 \
      -cuda-warmup 0 -cuda-runs 1 \
      >"$test_tmp/memory-dot.out" 2>&1
  if grep -F "T(init," "$test_tmp/memory.dot" >/dev/null; then
    echo "memory DAG unexpectedly contains scratch initialization tasks" >&2
    exit 1
  fi
  grep -F "T(memory_bound,(0,0,0))" \
    "$test_tmp/memory.dot" >/dev/null
  grep -F "S(0,0,0)(write)(64)" "$test_tmp/memory.dot" >/dev/null
fi

if command -v cuobjdump >/dev/null; then
  cuobjdump --dump-sass "$task_bench" >"$test_tmp/task-bench.sass"
  grep -F "LDG.E.U8" "$test_tmp/task-bench.sass" >/dev/null
  grep -F "STG.E.U8" "$test_tmp/task-bench.sass" >/dev/null
  grep -F "FFMA" "$test_tmp/task-bench.sass" >/dev/null
  grep -F "DFMA" "$test_tmp/task-bench.sass" >/dev/null
  grep -E "IMAD.*0x8b" "$test_tmp/task-bench.sass" >/dev/null
  grep -F "MEMBAR" "$test_tmp/task-bench.sass" >/dev/null
fi

echo "CUDASTF integration tests passed"
