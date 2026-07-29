# Task-Level Parallelism And Concurrency Metrics

## Research Objective

Determine how CUDASTF DAG structure, measured task duration, and the runtime
schedule affect execution performance. The implemented metric layers are:

- topology: DAG structure with unit-cost tasks;
- potential parallelism: an ideal ASAP model weighted by serialized task
  measurements;
- actual concurrency: task overlap observed in a normal CUDASTF execution.

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

Raw run JSON schema 3 retains configuration, environment provenance, expanded
DAGs, placement, measured performance samples, and all public task-profiler
fields. Analysis JSON schema 4 contains topology and reproducible derived
metrics.

Offline analysis recomputes topology, workload, execution, environment, and
raw-data hashes before deriving metrics. Per-task normal durations remain in
raw JSON and are not duplicated in concurrency analysis. GPU UUID remains raw
provenance but is excluded from same-model environment compatibility.

## Deferred Analysis

The current schema does not implement paired-run comparisons, task inflation,
effective parallelism, start delay, per-device concurrency or balance, scaling,
or device-operation metrics. Those require separate input pairing, clock
correlation, or operation-level design. They must not be inferred from the
current topology, potential-parallelism, or task-concurrency fields.
