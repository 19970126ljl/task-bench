# CUDASTF Task Bench backend

This backend executes Task Bench DAGs with CUDASTF. The stream executor uses
a fresh `stream_ctx` and CUDASTF-managed logical data for every warmup and
measured sample.

## Build

```sh
make -C cudastf
```

The default CUDA architecture is `native`. Override it when building for a
different target:

```sh
make -C cudastf CUDA_ARCH=sm_80
```

`CCCL_ROOT` defaults to `../../cccl` relative to this directory and can be
overridden. `NVCC`, `HOST_CC`, and `HOST_CXX` are also configurable.
The backend builds an isolated core library under `cudastf/.build` with the
selected host compilers, so existing core artifacts from other backends do
not introduce a mixed-compiler link.

JSON output uses the `cudastf-task-bench-results` format and records normalized
DAG, workload, and CUDA kernel launch configuration. It also records Task
Bench and CCCL revisions plus dirty state, build type, CUDA architecture
option and resolved target, and device compute capability. Setup and complete
sample lifecycle timings are recorded only under each sample's `diagnostics`
object.

## Run

```sh
./cudastf/task_bench \
  -steps 16 -width 64 -type stencil_1d -field 2 \
  -kernel compute_bound -iter 65536 \
  -cuda-blocks-per-task 32 -cuda-threads-per-block 128 \
  -cuda-compute-dtype fp32 \
  -cuda-device 0 -cuda-warmup 1 -cuda-runs 5 \
  -cuda-json result.json
```

CUDASTF-specific options:

```text
-cuda-device N
-cuda-context stream
-cuda-warmup N
-cuda-runs N
-cuda-json FILE
-cuda-blocks-per-task N
-cuda-threads-per-block N
-cuda-shmem-bytes-per-block N
-cuda-compute-dtype fp32|fp64
```

Task Bench options retain their existing meaning. `-and` submits multiple
expanded DAGs into the same sample context with separate logical data
stores. CUDA device, context, warmup, sample count, and JSON path are global.
Kernel launch options and compute data type apply to the current graph and
reset to their defaults after `-and`.
Dynamic shared-memory requests may use the selected device's opt-in per-block
limit; the backend configures the selected kernel specialization before
submission when the request exceeds the legacy limit.

`CUDASTF_DEFAULT_ALLOCATOR` may select `cached`, `cached_fifo`, `uncached`, or
`pooled`; it defaults to `cached`. This is a CUDASTF runtime setting rather
than a Task Bench option, and its effective value is recorded in results.

## GPU workloads

The backend supports `empty`, `busy_wait`, `memory_bound`, and
`compute_bound`. Every workload first reads every predecessor's complete
output and writes its own complete output:

- `empty` performs only that dependency-data transform.
- `busy_wait` performs the data transform once, followed by `-iter`
  deterministic modular integer updates.
- `memory_bound` performs the data transform once, then copies between regions
  of private scratch data `-iter` times.
- `compute_bound` performs the data transform once, followed by 64 fused
  multiply-add operations per `-iter`, or 128 FLOPs per iteration. Arithmetic
  uses FP32 or FP64.

`-iter` is the total logical work for one task, not work per CUDA thread.
Grid-stride partitioning distributes that fixed work across the configured
blocks and threads. Launch geometry therefore changes within-task parallelism
and resource demand without multiplying the requested operation count.

`-output` is the number of dependency-data bytes produced by each task. A task
with fan-in `F` reads `F * output_bytes` and writes `output_bytes`. Root tasks
have no input reads. These requested bytes describe logical task-data access;
they are not measurements of physical DRAM traffic or inter-GPU transfers.

`memory_bound` requests `-scratch` bytes of temporary CUDASTF logical data for
each task. `-sample N` divides that scratch into `N` equal samples. A
memory iteration selects sample
`(timestep * iterations + iteration) % samples`, reads its first half, and
writes its second half. The requested traffic per iteration is therefore
`scratch_bytes / samples`, split evenly between reads and writes. Scratch is
created during task submission with a first `.write()` access and is not
initialized because its values are not part of the benchmark result.

The user-side scratch handle is released after that task has acquired its
dependencies and enqueued its kernel. CUDASTF decides when the physical memory
can be reused and whether reuse adds runtime prerequisites. Allocation, reuse,
and any resulting waits are therefore part of the measured runtime behavior;
the backend does not predict scratch residency from the DAG task count.

Each CUDASTF task enqueues exactly one CUDA kernel. The kernel obtains every
logical-data pointer from `task.get()` immediately before launch, reads the
inputs, writes the output, and runs the selected workload. A memory task also
obtains its private scratch pointer in the same way. The default launch is 32
blocks of 128 threads. Use one block explicitly when measuring a minimal
kernel footprint.

## Visualize a DAG

Graphviz is required to render CUDASTF's DOT output. The visualization helper
runs exactly one measured sample, keeps the generated `.dot` file, and selects
the renderer from the output filename:

```sh
./cudastf/render_dag.sh dag.png -- \
  -steps 8 -width 16 -type stencil_1d -field 2
```

PDF and SVG output are also supported. SVG or PDF is generally easier to
inspect than PNG for large DAGs.

Task labels use `T(operation,(dag,step,point))`. Logical data labels use
`D(dag,field,point)` because Task Bench may reuse a field across multiple
timesteps.

CUDASTF prerequisite nodes are hidden by default. Include them when diagnosing
runtime operations with:

```sh
CUDASTF_DOT_IGNORE_PREREQS=0 \
  ./cudastf/render_dag.sh dag.svg -- \
  -steps 4 -width 8 -type no_comm
```

DOT tracing is diagnostic instrumentation and must not be enabled for
performance measurements.

## Metrics

- `Tasks` is the number of tasks in the expanded DAG.
- `Dependency edges` is the total number of predecessor relationships.
- `Task data accesses` counts one output write per task plus one read per
  predecessor, before CUDASTF merges accesses to the same logical data.
- `Task data store bytes` is the sum of dependency-data logical slot sizes.
- Modeled task-data and scratch bytes are requested logical accesses, not
  measurements of physical allocation, DRAM traffic, or transferred bytes.
- `Submission` is host wall time spent constructing and submitting CUDASTF
  tasks.
- `DAG makespan` is the CUDA event interval from the synchronized start
  boundary before submission to the final CUDASTF fence. It includes GPU idle
  gaps caused by dynamic submission and runtime scheduling.

## Configuration identity

Each graph has a `topology_hash` computed only from its ordered tasks and
predecessors. It remains stable when workload, data layout, launch geometry, or
compute data type changes.

The root `execution_config_hash` identifies the work submitted in one sample.
It includes ordered graph topologies, effective workload and data settings,
kernel launch configurations, context, device placement, and the logical-data
allocator selected by `CUDASTF_DEFAULT_ALLOCATOR`. The effective allocator is
also printed and stored in JSON. Sampling counts, output paths, measured
timings, software revisions, and hardware identity are recorded separately
and do not affect this hash.

## Current limits

- Only `stream_ctx` is supported.
- Supported kernels are `empty`, `busy_wait`, `memory_bound`, and
  `compute_bound`.
- `memory_bound` requires nonzero private scratch. Scratch must divide evenly
  into `-sample` samples, and each sample must split evenly into source and
  destination regions. Other workloads require `scratch=0`.
- A task may have at most 32 predecessors, or 33 task-data accesses including
  output. A memory task additionally declares one temporary scratch write.
- `field=1` is rejected if any task reads a different predecessor point. Use
  at least `field=2` for stencils and other cross-point dependencies.
- `random_spread` is rejected because Task Bench core does not implement it.
- Data pointers are obtained from `task.get()` immediately before launch
  and are never cached.

Run the GPU integration tests with:

```sh
make -C cudastf test
```
