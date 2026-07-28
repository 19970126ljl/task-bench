# Task-Level Parallelism Metrics

## Research Objective

Determine whether task-level DAG parallelism can explain or predict CUDASTF
execution performance, why potential parallelism and realized concurrency
differ, and when increasing task parallelism improves or harms performance.

The metric model has four semantic layers:

- topology: graph structure with unit-cost tasks;
- measured task duration: isolated task costs collected with global task
  serialization;
- runtime schedule: task behavior under normal CUDASTF execution;
- device operations: GPU and interconnect activity used for later root-cause
  analysis.

Task serialization and task profiling are independent controls. Serialization
changes scheduling. Profiling records task GPU activity. Potential DAG
parallelism requires both controls to be enabled; actual runtime concurrency
requires profiling with serialization disabled.

## Common Definitions

Each logical task has the stable key `(dag_index, timestep, point)`. A profiled
CUDASTF task is associated with that key through `(context_id, task_id)` and
validated against its symbol and configured device. Measurements must never
assume submission, profiler output, or completion order is stable.

For task `i`:

```text
pred(i)          DAG predecessor tasks
device(i)        configured placement device
submit_i         time the runtime receives the task
dag_ready_i      max completion time of pred(i), or execution start for a source
dag_eligible_i   max(submit_i, dag_ready_i)
start_i          beginning of the task's correlated GPU work
finish_i         completion of the task's correlated GPU work
duration_i       finish_i - start_i
```

The profiler interval is the envelope from the first through the last
correlated GPU operation in the task body. It excludes waiting for
predecessors and CUDASTF-managed acquisition before the task body. A task may
contain multiple GPU operations; its interval ends after its last operation.

Timestamp formulas are valid only within a common clock domain. Task and
device activity timestamps in one CUPTI profile share an origin. Host and
device timestamps, or unrelated GPU clock domains, must not be subtracted
without explicit correlation.

Duration distributions use mean, median, p95, and population coefficient of
variation:

```text
CV = population_standard_deviation / mean
```

Time-varying parallelism and concurrency use time-weighted statistics. A value
lasting 10 ms has ten times the weight of a value lasting 1 ms. Topology level
statistics instead give every unit-cost ASAP level equal weight.

Warmups are excluded from stored samples and all derived metrics. Measured
repetitions are independent. Multiple independent DAGs are reported both per
DAG and as a disjoint union; all DAGs begin at time zero in a combined model.

## Topology

Topology metrics use only tasks and dependency edges. Every task has unit cost;
workload, data size, placement, and measured time have no effect.

```text
start_i^0  = max(finish_j^0 for j in pred(i)), or 0 for a source
finish_i^0 = start_i^0 + 1

N             number of tasks
E             number of DAG dependency edges
L_0           unit-cost critical-path length
P_0           N / L_0, average topology parallelism
A_0(k)        tasks in unit-cost ASAP level k
A_0_peak      max A_0(k)
A_0_p50/p95   level-width quantiles
A_0_cv        variation of level widths
```

Every integer level in `[0, L_0)` contributes one observation, so `P_0` is the
arithmetic mean of `A_0(k)`. `E` describes the graph but is not itself a
parallelism metric. `A_0_peak` is the peak of this deterministic unit-cost ASAP
schedule, not maximum antichain width.

## Measured Task Duration

With global task serialization enabled, no two task GPU activity intervals may
overlap across any selected device. The original DAG, task body, inputs,
outputs, placement, and runtime-managed data movement remain unchanged; this
is not a separate microbenchmark.

For every logical task, collect one duration per measured serialized sample:

```text
d_i^measured = median serialized GPU activity duration for task i
```

The median is the selected aggregation. These are real measurements, not
estimates. The minimum is not used because it decreases systematically as the
number of repetitions increases. Variation between repetitions remains
measurement diagnostics; variation across `d_i^measured` values describes
logical task-time imbalance.

The serialized run's `dag_makespan_ms` is collection cost. It is neither
sequential DAG latency nor a performance baseline and is not used in the
parallelism model.

For multi-GPU placement, each task is measured on its configured device.
Serialization prevents overlap globally, not merely per device.

## Weighted DAG Parallelism

Assign `d_i^measured` to the corresponding DAG node and construct an
idealized ASAP schedule with unlimited task resources:

```text
start_i^weighted  = max(finish_j^weighted for j in pred(i)), or 0 for a source
finish_i^weighted = start_i^weighted + d_i^measured

W_measured        = sum d_i^measured
L_weighted        = max finish_i^weighted
P(t)              = count(i where start_i^weighted <= t < finish_i^weighted)
P_average         = W_measured / L_weighted
```

This schedule is derived from measured node weights; it is not an additional
executed run. It describes potential DAG parallelism under the assumption of
unlimited task resources.

Selected outputs are:

```text
W_measured              measured task work
L_weighted              weighted critical path
P_average               average DAG parallelism
P_peak                  peak DAG parallelism
P_p50/p95               time-weighted parallelism quantiles
P_cv                    temporal variation of parallelism
task duration stats     mean, median, p95, and CV of d_i^measured
```

The public analysis path is `derived_metrics.parallelism`. It is available
only when task profiling and task serialization are both enabled. Normal
profiled runs retain their raw task intervals but do not produce these metrics.

## Runtime Schedule

This layer executes the unchanged DAG with normal CUDASTF scheduling and task
serialization disabled. Profiling must not otherwise alter scheduling.

Two measurement modes remain distinct:

```text
T_exec(r)          DAG makespan with task profiling disabled
d_i^runtime(r)     profiled duration of task i
W_runtime(r)       sum d_i^runtime(r)
T_profile(r)       makespan of the same profiled run
C(t,r)             actual active task intervals at time t
```

`T_exec` is the authoritative end-to-end execution latency. `T_profile` and
all `C(t,r)` values come from one profiled execution and are never mixed with
timing from another sample.

For every profiled run:

```text
average C(t,r) = W_runtime(r) / T_profile(r)
integral C(t,r) over the trace = W_runtime(r)
```

The integral identity is a trace acceptance check. Selected concurrency
outputs are average, peak, p50, p95, and CV of `C(t,r)`, plus the distribution
of `d_i^runtime(r)`. These results will live at
`derived_metrics.concurrency`.

Performance comparisons use matching workload and environment identities:

```text
task completion rate    N / T_exec
effective parallelism   W_measured / T_exec
latency stretch         T_exec / L_weighted
parallelism efficiency  L_weighted / T_exec

task inflation_i        d_i^runtime / d_i^measured
work inflation          W_runtime / W_measured
work efficiency         W_measured / W_runtime
observed concurrency    W_runtime / T_profile
```

`W_runtime / T_profile` can increase when tasks slow down under contention.
`W_measured / T_exec` is therefore reported separately as useful measured task
work completed per unit of unprofiled execution time.

The first task waiting metric is:

```text
start_delay_i = start_i - dag_eligible_i
```

Selected scheduling metrics are start-delay median and p95, plus average and
peak DAG-eligible-but-not-started tasks. The term `start_delay` is deliberate:
it may include runtime data constraints, scheduler delay, and resource waiting.

## Multi-GPU Comparison

Per-device task metrics are:

```text
N_d                 tasks assigned to device d
W_measured,d        sum d_i^measured assigned to device d
W_runtime,d         sum d_i^runtime executed on device d
measured_balance    max(W_measured,d) / mean(W_measured,d)
runtime_balance     max(W_runtime,d) / mean(W_runtime,d)
remote_edges        edges crossing configured devices
remote_tasks        tasks with at least one remote predecessor
```

A balance ratio of 1 is even. Raw per-device values accompany every ratio so
idle or lightly loaded devices remain visible.

Scaling compares otherwise identical normal runs:

```text
speedup_G      T_exec,1GPU / T_exec,GPU
efficiency_G   speedup_G / G
```

`efficiency_G` is used only for homogeneous selected GPUs. Actual transfer
count, bytes, and overlap are not inferred from remote edges.

## Storage And Integrity

Raw run JSON retains configuration, environment provenance, expanded DAGs,
task placement, all measured samples, and all public task-profiler fields.
Derived analysis JSON contains topology and reproducible derived metrics.

Offline analysis recomputes topology, workload, execution, environment, and
raw-data hashes before deriving metrics. Serialized and normal runs may be
compared only when workload and environment hashes match. GPU UUID is retained
as provenance but excluded from same-model environment compatibility.

## Device Operations

Device-operation analysis is deferred until task-level results require deeper
explanation. Its intended scope includes GPU utilization, SM active time,
memory throughput, PCIe or NVLink throughput, kernel and copy duration,
operation concurrency, and computation/communication overlap.

Task intervals and GPU-operation intervals remain separate: one task may
launch multiple kernels or copies. Device-operation metrics must not be mixed
into topology, weighted DAG parallelism, or task concurrency definitions.
