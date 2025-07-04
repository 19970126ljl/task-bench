# CUDA STF Examples

This document provides practical examples for using the CUDA STF implementation in Task Bench.

## Basic Examples

### Example 1: Simple Task Execution

**Command**:
```bash
./main -steps 2 -width 2
```

**Output**:
```
Running Task Benchmark
  Configuration:
    Task Graph 1:
      Time Steps: 2
      Max Width: 2
      Dependence Type: trivial
      Kernel: empty
Total Tasks 4
Total Dependencies 0
Elapsed Time 1.31e-02 seconds
FLOP/s 0.000000e+00
```

**Explanation**: Executes 4 empty tasks (2 timesteps × 2 width) with no dependencies.

### Example 2: Compute-Bound Workload

**Command**:
```bash
./main -type stencil_1d -steps 3 -width 2 -kernel compute_bound -iter 1000
```

**Output**:
```
Running Task Benchmark
  Configuration:
    Task Graph 1:
      Time Steps: 3
      Max Width: 2
      Dependence Type: stencil_1d
      Kernel: compute_bound
      Iterations: 1000
Total Tasks 6
Total Dependencies 8
Total FLOPs 384000
Elapsed Time 1.71e-02 seconds
FLOP/s 2.24e+07
```

**Explanation**: 
- 6 tasks with 1D stencil dependencies
- Each task performs 1000 iterations of floating-point computation
- Achieves 22.4 MFLOP/s performance

### Example 3: Memory-Bound Workload

**Command**:
```bash
./main -type stencil_1d -steps 3 -width 2 -kernel memory_bound -iter 1000 -scratch 1024
```

**Output**:
```
Running Task Benchmark
  Configuration:
    Task Graph 1:
      Time Steps: 3
      Max Width: 2
      Dependence Type: stencil_1d
      Kernel: memory_bound
      Iterations: 1000
      Scratch Bytes: 1024
Total Tasks 6
Total Dependencies 8
Elapsed Time 1.73e-02 seconds
```

**Explanation**: Memory-intensive workload using 1024 bytes of scratch memory per task.

## Dependency Pattern Examples

### Trivial Pattern (Embarrassingly Parallel)

**Command**:
```bash
./main -type trivial -steps 3 -width 4 -kernel compute_bound -iter 500
```

**Use Case**: Independent tasks with no communication
**Characteristics**: 
- Maximum parallelism
- No synchronization overhead
- Ideal for GPU execution

### Stencil 1D Pattern

**Command**:
```bash
./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 2000
```

**Use Case**: Finite difference methods, signal processing
**Characteristics**:
- Each task depends on neighbors from previous timestep
- Moderate parallelism
- Common in scientific computing

### Tree Pattern

**Command**:
```bash
./main -type tree -steps 4 -width 8 -kernel compute_bound -iter 1000
```

**Use Case**: Hierarchical algorithms, reduction operations
**Characteristics**:
- Tree-like dependency structure
- Decreasing parallelism over time
- Common in divide-and-conquer algorithms

### All-to-All Pattern

**Command**:
```bash
./main -type all_to_all -steps 3 -width 4 -kernel compute_bound -iter 500
```

**Use Case**: Matrix operations, dense linear algebra
**Characteristics**:
- High communication overhead
- Complex dependency structure
- Challenging for parallelization

## Kernel Type Examples

### Empty Kernel (Overhead Measurement)

**Command**:
```bash
./main -type stencil_1d -steps 5 -width 10 -kernel empty
```

**Purpose**: Measure STF and Task Bench overhead
**Expected Results**: Very low execution time, 0 FLOP/s

### Compute-Bound Kernel

**Light Workload**:
```bash
./main -kernel compute_bound -iter 100 -steps 2 -width 2
```

**Heavy Workload**:
```bash
./main -kernel compute_bound -iter 10000 -steps 3 -width 4
```

**Characteristics**:
- FLOP/s increases with iteration count
- Good for measuring computational performance
- Minimal memory usage

### Memory-Bound Kernel

**Small Scratch**:
```bash
./main -kernel memory_bound -iter 500 -scratch 512 -steps 2 -width 2
```

**Large Scratch**:
```bash
./main -kernel memory_bound -iter 500 -scratch 4096 -steps 2 -width 2
```

**Characteristics**:
- Performance limited by memory bandwidth
- Scratch size affects memory access patterns
- Good for testing memory subsystem

### Busy-Wait Kernel

**Command**:
```bash
./main -kernel busy_wait -iter 1000 -steps 2 -width 4
```

**Purpose**: Simulate CPU-bound waiting or synchronization
**Characteristics**:
- Time-based performance measurement
- Simulates real-world waiting scenarios

### Load Imbalance Kernel

**No Imbalance**:
```bash
./main -kernel load_imbalance -iter 1000 -imbalance 0.0 -steps 2 -width 4
```

**Moderate Imbalance**:
```bash
./main -kernel load_imbalance -iter 1000 -imbalance 0.2 -steps 2 -width 4
```

**High Imbalance**:
```bash
./main -kernel load_imbalance -iter 1000 -imbalance 0.5 -steps 2 -width 4
```

**Expected Behavior**: FLOP count increases with imbalance factor

## Performance Benchmarking Examples

### Scalability Testing

**Small Scale**:
```bash
./main -type stencil_1d -steps 3 -width 4 -kernel compute_bound -iter 2000
```

**Medium Scale**:
```bash
./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 5000
```

**Large Scale**:
```bash
./main -type stencil_1d -steps 8 -width 16 -kernel compute_bound -iter 2000
```

**Expected Results**:
- FLOP/s should increase with scale
- Execution time may increase sub-linearly
- GPU utilization improves with larger workloads

### Iteration Scaling

**Low Iterations**:
```bash
./main -kernel compute_bound -iter 100 -steps 2 -width 2
```

**Medium Iterations**:
```bash
./main -kernel compute_bound -iter 1000 -steps 2 -width 2
```

**High Iterations**:
```bash
./main -kernel compute_bound -iter 10000 -steps 2 -width 2
```

**Expected Results**: FLOP/s should remain relatively constant, total FLOPs increase linearly

## Comparative Analysis Examples

### Kernel Performance Comparison

Run the same task graph with different kernels:

```bash
echo "Empty kernel:"
./main -type stencil_1d -steps 3 -width 4 -kernel empty

echo "Compute-bound kernel:"
./main -type stencil_1d -steps 3 -width 4 -kernel compute_bound -iter 1000

echo "Memory-bound kernel:"
./main -type stencil_1d -steps 3 -width 4 -kernel memory_bound -iter 1000

echo "Busy-wait kernel:"
./main -type stencil_1d -steps 3 -width 4 -kernel busy_wait -iter 1000
```

### Dependency Pattern Comparison

Run the same kernel with different patterns:

```bash
echo "Trivial pattern:"
./main -type trivial -steps 3 -width 4 -kernel compute_bound -iter 1000

echo "Stencil 1D pattern:"
./main -type stencil_1d -steps 3 -width 4 -kernel compute_bound -iter 1000

echo "Tree pattern:"
./main -type tree -steps 3 -width 4 -kernel compute_bound -iter 1000
```

## Advanced Examples

### Complex Task Graph

**Command**:
```bash
./main -type stencil_1d -steps 10 -width 32 -kernel compute_bound -iter 5000
```

**Characteristics**:
- 320 total tasks
- Complex dependency structure
- High computational workload
- Tests STF scalability

### Mixed Workload Simulation

Create a script to test multiple scenarios:

```bash
#!/bin/bash
echo "=== CUDA STF Performance Suite ==="

echo "1. Overhead measurement:"
./main -type trivial -steps 5 -width 10 -kernel empty

echo "2. Compute performance:"
./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 5000

echo "3. Memory performance:"
./main -type stencil_1d -steps 3 -width 6 -kernel memory_bound -iter 1000 -scratch 2048

echo "4. Load imbalance test:"
./main -type stencil_1d -steps 3 -width 6 -kernel load_imbalance -iter 2000 -imbalance 0.3

echo "5. Complex dependency test:"
./main -type all_to_all -steps 3 -width 4 -kernel compute_bound -iter 1000
```

### Performance Profiling

For detailed performance analysis:

```bash
# Profile with NVIDIA Nsight Compute
ncu --set full ./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 5000

# Profile with NVIDIA Nsight Systems
nsys profile ./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 5000
```

## Expected Performance Ranges

### Typical FLOP/s Results

Based on test system performance:

- **Empty Kernel**: 0 FLOP/s (overhead measurement)
- **Compute-Bound**: 10^7 to 10^9 FLOP/s (depends on iteration count)
- **Memory-Bound**: Limited by memory bandwidth
- **Busy-Wait**: 0 FLOP/s (time-based measurement)
- **Load Imbalance**: Variable based on imbalance factor

### Execution Time Ranges

- **Small Tasks** (< 10 tasks): 10-20 ms
- **Medium Tasks** (10-100 tasks): 20-50 ms  
- **Large Tasks** (> 100 tasks): 50+ ms

### Scalability Expectations

- **Task Count**: Linear scaling up to GPU capacity
- **Iteration Count**: Linear FLOP scaling, constant FLOP/s
- **Dependency Complexity**: May reduce parallelism efficiency

## Troubleshooting Examples

### Low Performance

If FLOP/s is unexpectedly low:

```bash
# Increase workload size
./main -kernel compute_bound -iter 10000 -steps 5 -width 8

# Check GPU utilization with nvidia-smi during execution
nvidia-smi -l 1 &
./main -kernel compute_bound -iter 5000 -steps 8 -width 16
```

### Memory Issues

If encountering memory errors:

```bash
# Reduce task graph size
./main -steps 2 -width 2 -kernel compute_bound -iter 1000

# Reduce scratch memory
./main -kernel memory_bound -scratch 256 -iter 500
```

### Dependency Issues

If results seem incorrect:

```bash
# Test with simple patterns first
./main -type trivial -kernel compute_bound -iter 1000
./main -type no_comm -kernel compute_bound -iter 1000
```
