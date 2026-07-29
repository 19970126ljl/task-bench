# Task-Level Parallelism, Capacity, And Concurrency Metrics

## Research Objective

Determine how CUDASTF DAG structure, measured task duration, and the runtime
schedule affect execution performance. The implemented metric layers are:

- topology: DAG structure with unit-cost tasks;
- kernel capacity: occupancy-based block residency and the runnable task count
  needed to fill it;
- potential parallelism: an ideal ASAP model weighted by serialized task
  measurements;
- actual concurrency: task overlap observed in a normal CUDASTF execution;
- cross-mode comparison: task efficiency, concurrency versus parallelism, and
  ideal versus actual execution time for compatible serialized and normal
  profiled runs.

Task profiling and task serialization are independent controls. Potential
parallelism requires both controls enabled. Actual concurrency requires task
profiling enabled and task serialization disabled.

## Task Identity And Time

Each logical task has the stable key `(dag_index, timestep, point)`. A profiled
CUDASTF task is associated with that key through `(context_id, task_id)` and is
validated against its symbol and configured device. Submission, profiler
output, and completion order are not identity contracts.

For task `i`, the profiler interval is:

```text
start_i       earliest correlated GPU operation start
finish_i      latest correlated GPU operation end
duration_i    finish_i - start_i
```

The interval is the task GPU activity envelope. It excludes predecessor
waiting and CUDASTF-managed acquisition before the task body. One task may
contain multiple correlated GPU operations.

Task timestamps within one CUPTI profile share the profile's CUPTI origin.
CUDA event durations, CUPTI timestamps, and host timestamps must not be
treated as having a common origin without explicit correlation.

Duration distributions use mean, median, nearest-rank p95, and population
coefficient of variation. Time-varying metrics use duration-weighted p50,
p95, and population coefficient of variation. Intervals are half-open:
`[start, finish)`.

Warmups are never stored or included in derived metrics. Every measured sample
is analyzed independently.

## Topology

Topology metrics use only task and dependency-edge structure. Each task has
unit cost:

```text
start_i^0  = max(finish_j^0 for j in pred(i)), or 0 for a source
finish_i^0 = start_i^0 + 1

N             number of tasks
E             number of dependency edges
L_0           unit-cost critical-path length
P_0           N / L_0
A_0(k)        tasks in unit-cost ASAP level k
```

Selected outputs are task count, edge count, critical-path length, and the
average, peak, p50, p95, and CV of ASAP level width. Level percentiles weight
each integer ASAP level equally. Combined topology treats independent DAGs as
a disjoint union beginning at modeled time zero.

## Kernel Occupancy Capacity

CUDA calculates `max_active_blocks_per_sm` for each compiled DAG kernel and
assigned device using the configured threads per block and dynamic shared
memory. The occupancy API incorporates register, shared-memory, thread, warp,
and block residency limits. With device SM count `S_d`, active blocks per SM
`A_d`, and task grid size `B`:

```text
resident_blocks_d            = S_d * A_d
occupancy_saturation_tasks_d = ceil(resident_blocks_d / B)
```

Only devices assigned at least one task from that DAG are included. Multi-GPU
combined capacity sums per-device resident blocks and sums the per-device task
counts after rounding because one task grid cannot span devices. Different
DAG kernels are reported separately and are not combined into a mixed-kernel
capacity.

The public path is `kernel_capacity`. It is available independently of task
profiling and serialization. `occupancy_saturation_tasks` is the number of
runnable task kernels needed to supply enough blocks to fill theoretical block
residency. It is neither maximum task concurrency nor performance saturation,
and actual task-envelope concurrency may be above or below it.

## Potential DAG Parallelism

With global task serialization enabled, task GPU envelopes must not overlap on
any selected device. For every logical task:

```text
d_i^serialized = median task duration across measured serialized samples
```

These are real profiler measurements made without inter-task GPU overlap. The
serialized sample's `dag_makespan_ms` is collection cost, not sequential DAG
latency or a performance baseline.

Map each `d_i^serialized` back to its DAG node and construct an ideal ASAP
schedule with unlimited task resources:

```text
start_i^weighted  = max(finish_j^weighted for j in pred(i)), or 0
finish_i^weighted = start_i^weighted + d_i^serialized

task_work         = sum d_i^serialized
critical_path     = max finish_i^weighted
P(t)              = count(i where start_i^weighted <= t < finish_i^weighted)
average           = task_work / critical_path
```

Selected outputs are `task_work_ms`, `critical_path_ms`, task-duration
statistics, and the average, peak, p50, p95, and CV of `P(t)`. This is a
measured-weight DAG model, not a real execution trace and not a pure topology
property.

The public path is `derived_metrics.parallelism`. It is available only with
profiling and serialization enabled. Multiple DAGs are also evaluated as an
ideal disjoint union beginning at modeled time zero.

## Actual Task Concurrency

Normal execution leaves task serialization disabled. Profiling records the
unchanged CUDASTF schedule without otherwise selecting a different execution
mode.

For every measured sample and task set:

```text
task_work       = sum duration_i
GPU_span        = max(finish_i) - min(start_i)
C(t)            = count(i where start_i <= t < finish_i)
GPU_average     = task_work / GPU_span
```

`Task GPU span` begins at the first profiled task GPU operation and ends at the
last. Its concurrency distribution includes internal periods with no active
task. Average concurrency may therefore be less than one.

Each DAG receives its own Task GPU span. Combined concurrency preserves all
DAGs on their real CUPTI timeline, so overlap between DAGs is retained.

The combined result also has an `End-to-end span`:

```text
end_to_end_span       = dag_makespan_ms from the same profiled sample
extra_zero_time       = end_to_end_span - GPU_span
end_to_end_average    = task_work / end_to_end_span
```

`dag_makespan_ms` is a CUDA event duration from the synchronized boundary
before submission through the final CUDASTF fence. It is not timestamp-aligned
with CUPTI. The fence makes it an enclosing execution duration; aggregate
time-weighted statistics require only the total `extra_zero_time`, which is
added to the zero-concurrency bucket.

If End-to-end span is non-positive or shorter than Task GPU span, that sample's
End-to-end result is unavailable. Its Task GPU result remains valid. There is
no per-DAG End-to-end span because the current measurement provides only one
boundary for the complete sample.

Selected per-sample outputs are `task_work_ms`, task-duration statistics, and
complete concurrency statistics for both spans where available. The public
path is `derived_metrics.concurrency`. It stores every measured sample, then
reports the across-sample median and nearest-rank p95 of every scalar. Sample
timelines are never merged.

## Storage And Integrity

Raw run JSON schema 4 retains configuration, environment provenance, expanded
DAGs, placement, measured performance samples, and all public task-profiler
fields. Analysis JSON schema 5 contains topology, kernel capacity, and
reproducible derived metrics.

Offline analysis recomputes topology, workload, execution, environment, and
raw-data hashes before deriving metrics. Device SM count participates in the
environment hash, while compiled kernel resource and occupancy fields
participate in the raw-data hash. Per-task normal durations remain in raw JSON
and are not duplicated in concurrency analysis. GPU UUID remains raw provenance
but is excluded from same-model environment compatibility.

## Cross-Mode Comparison

Cross-mode metrics use two independent raw run JSON inputs: one profiled run
with task serialization enabled and one profiled normal run. Both inputs are
first validated by the regular offline analyzer. Pairing additionally requires
equal workload and environment hashes, DAG order and topology, logical task
keys and configured devices, and kernel capacity. Measured sample counts may
differ.

Serialized observations produce one median duration per logical task. No
serialized sample is paired with a normal sample. For each normal sample:

```text
d_i^serialized = median serialized GPU-envelope duration for task i
d_i^normal     = GPU-envelope duration for task i in this normal sample

W_serialized   = sum d_i^serialized
W_normal       = sum d_i^normal
T_ideal        = ideal ASAP DAG time using d_i^serialized
P              = W_serialized / T_ideal
C_gpu          = W_normal / Task_GPU_span
C_end_to_end   = W_normal / End_to_end_span

task_duration_efficiency_i = d_i^serialized / d_i^normal
task_work_efficiency       = W_serialized / W_normal
task_gpu_span.concurrency_to_parallelism = C_gpu / P
end_to_end_span.concurrency_to_parallelism = C_end_to_end / P
task_gpu_span.ideal_to_actual_time = T_ideal / Task_GPU_span
end_to_end_span.ideal_to_actual_time = T_ideal / End_to_end_span
```

Normal execution contributes measured durations, spans, and concurrency only;
it does not create a modeled normal critical path. The two decompositions are:

```text
task_gpu_span.ideal_to_actual_time =
    task_work_efficiency * task_gpu_span.concurrency_to_parallelism
end_to_end_span.ideal_to_actual_time =
    task_work_efficiency * end_to_end_span.concurrency_to_parallelism
```

The public comparison schema uses `task_work.efficiency`,
`span.concurrency_to_parallelism`, and `span.ideal_to_actual_time` for these
ratios. Per-DAG Task GPU spans and the combined Task GPU span use integer CUPTI
timestamps from each normal sample. End-to-end results are combined-only. If
any sample has an unavailable End-to-end span, its Task GPU metrics remain
valid but the End-to-end summary is unavailable.

Every normal sample retains per-DAG and combined task work, ideal time, average
parallelism, actual span time, average concurrency, and the two higher-is-
better ratios. Across-sample summaries report median and nearest-rank p95.
Task details retain the stable task key and configured device, serialized
median duration, normal duration median/p95, and duration-efficiency median/p95.
Ratios are not clamped.

The standalone command is:

```sh
python3 cudastf/compare.py \
  --serialized serialized-run.json \
  --normal normal-run.json \
  --output comparison.json
```

The output uses format `cudastf-task-bench-comparison` and schema 1. It records
both source raw-data hashes but does not modify or replace either raw run or its
single-run analysis.

## Deferred Analysis

The current schemas do not implement span stretch, effective parallelism,
start delay, per-device interference or balance, scaling, utilization, or
device-operation metrics. Those require additional semantic design, clock
correlation, or operation-level data. They must not be inferred from the
current topology, potential-parallelism, task-concurrency, or cross-mode fields.
