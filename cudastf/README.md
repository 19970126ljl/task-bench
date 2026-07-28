# CUDASTF Task Bench backend

This backend executes Task Bench DAGs with CUDASTF `stream_ctx`; every Task
Bench task submits one CUDA kernel. A sample uses one context across all
selected GPUs, and CUDASTF manages logical-data instances and migration.

## Build

```sh
make -C cudastf
```

The default CUDA architecture is `native`. Override it when needed:

```sh
make -C cudastf CUDA_ARCH=sm_80
```

`CCCL_ROOT` defaults to `../../cccl`. `NVCC`, `HOST_CC`, and `HOST_CXX` may
also be overridden. The CUPTI root is resolved from `CUDA_PATH`, `CUDA_HOME`,
or the selected `nvcc`; set `CUPTI_ROOT` explicitly for nonstandard CUDA
installations.

## Run

```sh
./cudastf/task_bench \
  -steps 16 -width 64 -type stencil_1d -field 2 \
  -kernel compute_bound -iter 65536 \
  -cuda-blocks-per-task 32 -cuda-threads-per-block 128 \
  -cuda-compute-dtype fp32 \
  -cuda-devices 0,1 -cuda-placement block \
  -cuda-warmup 1 -cuda-runs 5 \
  -cuda-json run.json -cuda-analysis-json analysis.json
```

CUDASTF options:

```text
-cuda-device N
-cuda-devices N,N,...
-cuda-placement block|cyclic
-cuda-context stream
-cuda-warmup N
-cuda-runs N
-cuda-task-serialization disabled|enabled
-cuda-task-profiler disabled|enabled
-cuda-json FILE
-cuda-analysis-json FILE
-cuda-blocks-per-task N
-cuda-threads-per-block N
-cuda-shmem-bytes-per-block N
-cuda-compute-dtype fp32|fp64
```

Run `./cudastf/task_bench -h` for Task Bench options and current defaults.
Launch options and compute data type apply to the current DAG and reset after
`-and`; task placement and sampling options are global. `block` assigns
contiguous point ranges and `cyclic` assigns points round-robin. The singular
and plural device options cannot be combined, and the default is GPU 0.
`CUDASTF_DEFAULT_ALLOCATOR`
selects `cached`, `cached_fifo`, `uncached`, or `pooled` and defaults to
`cached`.

Task serialization and profiling are independent and default to `disabled`.
Serialization calls CUDASTF `set_task_serialization()` and changes scheduling;
profiling collects task GPU activity without changing that setting. In
particular, normal profiled runs provide task traces, while profiled and
serialized runs provide isolated measured task durations.

## Workloads

Supported kernels:

- `empty`: dependency-data reads and output write only;
- `busy_wait`: deterministic integer work;
- `compute_bound`: FP32 or FP64 throughput work;
- `memory_bound`: copies within per-task CUDASTF logical scratch data.

Every task reads each predecessor's complete `-output` buffer and writes one
complete output buffer. `-iter` is total logical work per task, not work per
CUDA thread. `-imbalance` deterministically scales task iterations for
`busy_wait`, `compute_bound`, and `memory_bound`.

For `memory_bound`, `-scratch` is the allocated bytes per task and `-sample N`
divides it into per-iteration windows:

```text
scratch_window_bytes = scratch_bytes / samples
read_bytes_per_iteration = write_bytes_per_iteration
                         = scratch_window_bytes / 2
```

Exact kernel behavior and task iteration selection are defined in
[`workload.cuh`](workload.cuh) and [`workload.cc`](workload.cc). CUDASTF task
submission and data ownership are defined in [`main.cu`](main.cu).

## Results

Console output includes execution timing and derived topology metrics.
`-cuda-json` records configuration, the complete expanded DAG, placement,
identities, raw samples, and all public CUDASTF profiler fields.
`-cuda-analysis-json` writes topology and derived metrics separately. Warmups
are profiled when requested but are never recorded or used for metric
derivation.

`measured_duration_ms` uses the same profiler interval as a normal task trace:
the envelope from the first through the last correlated GPU operation in the
task body. For each logical task, it is the median across measured serialized
runs. These durations are mapped back to DAG nodes to derive weighted critical
path and potential DAG parallelism under a model with unlimited task resources.
CUDASTF-managed acquisition and automatic data movement before the task body
are serialized but are outside this profiler interval.

The serialized run's `dag_makespan_ms` is the cost of collecting isolated task
measurements, not a performance baseline. Normal runs continue to report it as
the end-to-end DAG makespan.

Analysis can be regenerated without a GPU:

```sh
python3 cudastf/analyze.py --input run.json --output analysis.json
```

The analyzer recomputes and validates topology, workload, execution,
environment, and raw-data hashes before deriving metrics. Analysis output uses
`derived_metrics.parallelism`; normal profiled traces leave that entry
unavailable and are retained for later `derived_metrics.concurrency` analysis.
GPU UUIDs remain in raw provenance but do not affect same-model environment
compatibility.

`DAG makespan` is the CUDA event interval from the synchronized boundary before
submission to the final CUDASTF fence. JSON field definitions are emitted by
[`results.cc`](results.cc); topology and execution identities are defined in
[`identity.cc`](identity.cc).

## Visualize

Graphviz is required:

```sh
./cudastf/render_dag.sh dag.png -- \
  -steps 8 -width 16 -type stencil_1d -field 2
```

PNG, SVG, and PDF are supported. DOT tracing is diagnostic instrumentation and
must not be enabled for performance measurements.

## Limits

- Only `stream_ctx` is supported; one sample uses one context for all devices.
- A task may have at most 32 predecessors.
- `memory_bound` requires nonzero scratch divisible into even-sized samples;
  other workloads require `scratch=0`.
- Cross-point dependencies require at least `field=2`.
- `random_spread` is unavailable because Task Bench core does not implement it.

Run all backend tests with:

```sh
make -C cudastf test
```
