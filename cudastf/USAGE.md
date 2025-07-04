# CUDA STF Usage Guide

This guide provides detailed usage instructions for the CUDA STF implementation in Task Bench.

## Installation and Setup

### Environment Setup

1. **Set Required Environment Variables**:
```bash
export USE_CUDASTF=1
export CCCL_DIR=/root/stf_exp/cccl
```

2. **Verify CUDA Installation**:
```bash
nvcc --version
nvidia-smi
```

3. **Build the Implementation**:
```bash
cd /root/stf_exp/task-bench
./build_all.sh
```

Or build directly:
```bash
cd cudastf
make clean && make
```

## Basic Usage Examples

### 1. Simple Execution Test

Test basic functionality:
```bash
./main -steps 2 -width 2
```

Expected output:
```
Running Task Benchmark
  Configuration:
    Task Graph 1:
      Time Steps: 2
      Max Width: 2
      ...
Total Tasks 4
Elapsed Time 1.31e-02 seconds
```

### 2. Dependency Pattern Testing

Test different dependency patterns:

**Trivial Pattern (No Dependencies)**:
```bash
./main -type trivial -steps 3 -width 3
```

**1D Stencil Pattern**:
```bash
./main -type stencil_1d -steps 4 -width 4
```

**Tree Pattern**:
```bash
./main -type tree -steps 3 -width 8
```

### 3. Kernel Type Testing

Test different computational kernels:

**Empty Kernel (Overhead Measurement)**:
```bash
./main -kernel empty -steps 2 -width 2
```

**Compute-Bound Kernel**:
```bash
./main -kernel compute_bound -iter 1000 -steps 3 -width 2
```

**Memory-Bound Kernel**:
```bash
./main -kernel memory_bound -iter 500 -scratch 1024 -steps 2 -width 2
```

**Busy-Wait Kernel**:
```bash
./main -kernel busy_wait -iter 1000 -steps 2 -width 2
```

**Load Imbalance Kernel**:
```bash
./main -kernel load_imbalance -iter 1000 -imbalance 0.1 -steps 3 -width 3
```

## Advanced Usage

### Performance Benchmarking

**Small Scale Benchmark**:
```bash
./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 5000
```

**Medium Scale Benchmark**:
```bash
./main -type stencil_1d -steps 8 -width 16 -kernel compute_bound -iter 2000
```

**Large Scale Benchmark**:
```bash
./main -type stencil_1d -steps 10 -width 32 -kernel compute_bound -iter 1000
```

### Comprehensive Testing

**Test All Dependency Patterns**:
```bash
for pattern in trivial no_comm stencil_1d tree dom fft all_to_all nearest spread random_nearest; do
    echo "Testing $pattern pattern:"
    ./main -type $pattern -steps 3 -width 4 -kernel compute_bound -iter 100
done
```

**Test All Kernel Types**:
```bash
for kernel in empty compute_bound memory_bound busy_wait compute_bound2 load_imbalance; do
    echo "Testing $kernel kernel:"
    ./main -type stencil_1d -steps 2 -width 2 -kernel $kernel -iter 100
done
```

### Memory and Scratch Testing

**Memory-Bound with Different Scratch Sizes**:
```bash
./main -kernel memory_bound -scratch 512 -iter 100
./main -kernel memory_bound -scratch 1024 -iter 100
./main -kernel memory_bound -scratch 2048 -iter 100
```

### Load Imbalance Testing

**Different Imbalance Factors**:
```bash
./main -kernel load_imbalance -imbalance 0.0 -iter 1000  # No imbalance
./main -kernel load_imbalance -imbalance 0.1 -iter 1000  # 10% imbalance
./main -kernel load_imbalance -imbalance 0.5 -iter 1000  # 50% imbalance
```

## Command Line Reference

### Required Parameters
- None (all parameters have defaults)

### Optional Parameters

**Task Graph Configuration**:
- `-steps N`: Number of timesteps (default: 1)
- `-width N`: Maximum width of task graph (default: 1)
- `-type PATTERN`: Dependency pattern (default: trivial)

**Kernel Configuration**:
- `-kernel TYPE`: Kernel type (default: empty)
- `-iter N`: Iterations for compute kernels (default: 0)
- `-scratch N`: Scratch memory size in bytes (default: 0)
- `-imbalance F`: Load imbalance factor 0.0-1.0 (default: 0.0)

**Performance Options**:
- `-samples N`: Number of timing samples (default: 16)

**Output Configuration**:
- `-output N`: Output bytes per task (default: 16)

### Dependency Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `trivial` | No dependencies | Embarrassingly parallel |
| `no_comm` | Sequential only | Pipeline processing |
| `stencil_1d` | 1D stencil | Finite difference methods |
| `tree` | Tree structure | Hierarchical algorithms |
| `dom` | Domain decomposition | Scientific computing |
| `fft` | FFT-like | Signal processing |
| `all_to_all` | All-to-all | Matrix operations |
| `nearest` | Nearest neighbor | Grid computations |
| `spread` | Spreading pattern | Broadcasting |
| `random_nearest` | Random neighbors | Graph algorithms |

### Kernel Types

| Kernel | Description | Parameters |
|--------|-------------|------------|
| `empty` | Minimal overhead | None |
| `compute_bound` | FP operations | `-iter` |
| `memory_bound` | Memory access | `-iter`, `-scratch` |
| `busy_wait` | Busy waiting | `-iter` |
| `compute_bound2` | Alternative compute | `-iter` |
| `load_imbalance` | Imbalanced work | `-iter`, `-imbalance` |

## Performance Analysis

### Understanding Output

**Basic Metrics**:
```
Total Tasks 40              # Number of tasks executed
Total Dependencies 88       # Number of dependencies
Total FLOPs 25602560       # Floating-point operations
Elapsed Time 3.98e-02 s    # Wall-clock time
FLOP/s 6.43e+08           # Performance metric
```

**Performance Interpretation**:
- Higher FLOP/s indicates better computational performance
- Lower elapsed time for same workload indicates better efficiency
- Dependency count affects parallelization potential

### Optimization Tips

1. **Kernel Selection**: Choose appropriate kernel for workload type
2. **Iteration Count**: Balance between accuracy and runtime
3. **Task Graph Size**: Larger graphs may show better GPU utilization
4. **Memory Management**: Use appropriate scratch sizes for memory-bound kernels

## Troubleshooting

### Build Issues

**CCCL Not Found**:
```bash
# Verify CCCL_DIR is set correctly
echo $CCCL_DIR
ls $CCCL_DIR/cudax/include
```

**CUDA Compilation Errors**:
```bash
# Check CUDA installation
nvcc --version
which nvcc
```

### Runtime Issues

**GPU Memory Errors**:
```bash
# Check GPU memory
nvidia-smi
# Reduce task graph size or iterations
./main -steps 2 -width 2 -iter 100
```

**Performance Issues**:
```bash
# Check GPU utilization
nvidia-smi -l 1
# Increase workload size
./main -steps 5 -width 8 -iter 5000
```

### Debug Mode

Enable debug output by modifying Makefile:
```makefile
NVCC_FLAGS += -DDEBUG -g -G
```

Then rebuild and run with cuda-gdb:
```bash
make clean && make
cuda-gdb ./main
```

## Best Practices

1. **Start Small**: Begin with small task graphs and increase size gradually
2. **Verify Correctness**: Test with known patterns before performance runs
3. **Monitor Resources**: Watch GPU memory and utilization during runs
4. **Consistent Environment**: Use same CUDA version and GPU for comparisons
5. **Multiple Runs**: Average results over multiple runs for stability

## Integration with Task Bench

The CUDA STF implementation integrates seamlessly with the Task Bench ecosystem:

- **Build System**: Controlled by `USE_CUDASTF` environment variable
- **Interface Compatibility**: Implements standard Task Bench App interface
- **Output Format**: Compatible with Task Bench analysis tools
- **Performance Metrics**: Standard FLOP/s and timing measurements
