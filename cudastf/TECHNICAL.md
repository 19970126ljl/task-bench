# CUDA STF Technical Documentation

This document provides detailed technical information about the CUDA STF implementation for Task Bench.

## Architecture Overview

### Core Components

```
CudaSTFApp
├── STF Context Management
├── Data Lifecycle Management  
├── Task Execution Engine
├── Dependency Resolution
└── Performance Measurement
```

### Class Hierarchy

```cpp
class CudaSTFApp : public App {
private:
    cudax::stf::context ctx;                    // STF execution context
    std::vector<char> host_data;                // Host data buffer
    cudax::stf::logical_data<slice<char>> data; // STF logical data
    
public:
    void execute_task_with_deps(long timestep, long point, 
                               const std::vector<std::pair<long, long>>& deps) override;
    void execute_task_no_deps(long timestep, long point) override;
};
```

## STF Integration Details

### Context Initialization

```cpp
// STF context setup with proper device management
cudax::stf::context ctx;
ctx.set_allocator<cudax::stf::data_place::device>(
    cudax::stf::par_allocator(cudax::mr::device_memory_resource{})
);
```

### Data Management Pattern

The implementation follows STF best practices for data lifecycle:

1. **Host Buffer Initialization**:
```cpp
std::vector<char> host_data(output_bytes, 0);
auto data = ctx.logical_data(host_data.data(), host_data.size());
```

2. **Task Submission with Dependencies**:
```cpp
ctx.task(data.rw()).apply([=] __device__ (auto output) {
    // Kernel execution
});
```

3. **Dependency Specification**:
- `.read()`: Read-only access
- `.write()`: Write-only access  
- `.rw()`: Read-write access

### Asynchronous Execution Model

STF provides automatic dependency resolution and asynchronous execution:

```cpp
// Tasks are submitted to STF context
ctx.task(data.rw()).apply(kernel_lambda);

// STF handles:
// - Dependency analysis
// - Memory transfers
// - Kernel scheduling
// - Synchronization
```

## CUDA Kernel Implementations

### Compute-Bound Kernel

```cpp
__global__ void compute_kernel(slice<char> output, long iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.size()) {
        float result = 0.0f;
        for (long i = 0; i < iterations; i++) {
            // Floating-point intensive computation
            result += sinf(cosf(float(i + idx))) * tanf(float(i));
        }
        output[idx] = static_cast<char>(result);
    }
}
```

**Characteristics**:
- High arithmetic intensity
- Minimal memory access
- Scales with iteration count
- FLOP count: `iterations * output.size() * 64`

### Memory-Bound Kernel

```cpp
__global__ void memory_kernel(slice<char> output, slice<char> scratch, 
                             long iterations, long timestep) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.size()) {
        for (long i = 0; i < iterations; i++) {
            // Memory-intensive operations
            int scratch_idx = (idx + i + timestep) % scratch.size();
            output[idx] = scratch[scratch_idx] + static_cast<char>(i);
        }
    }
}
```

**Characteristics**:
- High memory bandwidth utilization
- Scratch buffer access patterns
- Timestep-dependent addressing
- Memory throughput dependent

### Busy-Wait Kernel

```cpp
__global__ void busy_wait_kernel(slice<char> output, long iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.size()) {
        volatile long counter = 0;
        for (long i = 0; i < iterations; i++) {
            counter++; // Busy waiting simulation
        }
        output[idx] = static_cast<char>(counter % 256);
    }
}
```

**Characteristics**:
- Simulates CPU-bound waiting
- Minimal useful computation
- Time-based performance measurement

### Load Imbalance Kernel

```cpp
__global__ void compute2_kernel(slice<char> output, long iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output.size()) {
        // Variable work based on thread index
        long actual_iterations = iterations * (1.0f + imbalance * (idx % 2));
        float result = 0.0f;
        for (long i = 0; i < actual_iterations; i++) {
            result += sqrtf(float(i + idx));
        }
        output[idx] = static_cast<char>(result);
    }
}
```

**Characteristics**:
- Simulates real-world load imbalance
- Variable work per thread
- Configurable imbalance factor

## Dependency Management

### STF Dependency Model

STF automatically handles dependencies through data access patterns:

```cpp
// Task A writes data
ctx.task(data.write()).apply(kernel_a);

// Task B reads data (automatically waits for A)
ctx.task(data.read()).apply(kernel_b);

// Task C modifies data (waits for B)
ctx.task(data.rw()).apply(kernel_c);
```

### Task Graph Translation

Task Bench dependency patterns are translated to STF dependencies:

1. **Analyze Task Graph**: Extract dependencies from Task Bench graph
2. **Create Logical Data**: One logical data per task output
3. **Submit Tasks**: With appropriate read/write specifications
4. **STF Resolution**: Automatic dependency resolution and scheduling

### Multi-Task Dependencies

For tasks with multiple dependencies:

```cpp
void execute_task_with_deps(long timestep, long point, 
                           const std::vector<std::pair<long, long>>& deps) {
    // Create task with multiple input dependencies
    auto task = ctx.task(data.rw());
    
    // STF handles multiple dependencies automatically
    task.apply([=] __device__ (auto output) {
        // Kernel execution with all dependencies satisfied
    });
}
```

## Performance Characteristics

### FLOP Calculation

Different kernels have different FLOP counts:

```cpp
// Compute-bound kernel
long flops_per_task = iterations * output_bytes * 64;

// Compute2 kernel  
long flops_per_task = iterations * output_bytes * 4;

// Load imbalance kernel
long base_flops = iterations * output_bytes * 64;
long imbalance_flops = base_flops * (1.0 + imbalance * 0.5);
```

### Memory Usage

STF manages memory efficiently:

- **Host Buffers**: Initial data storage
- **Device Memory**: Automatic allocation by STF
- **Scratch Buffers**: Additional memory for memory-bound kernels
- **Reference Counting**: Automatic cleanup

### Scalability Analysis

Performance scales with:

1. **Task Graph Size**: More parallelism with larger graphs
2. **Iteration Count**: Higher arithmetic intensity
3. **GPU Utilization**: Better performance with sufficient work
4. **Memory Bandwidth**: Critical for memory-bound kernels

## Build System Integration

### Makefile Structure

```makefile
# Environment variable detection
CCCL_DIR ?= /root/stf_exp/cccl

# STF-specific flags
NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda
NVCC_FLAGS += -I$(CCCL_DIR)/cudax/include
NVCC_FLAGS += -I$(CCCL_DIR)/libcudacxx/include

# Architecture-specific compilation
NVCC_FLAGS += -arch=sm_86

# Optimization flags
NVCC_FLAGS += -O3
```

### Conditional Compilation

Integration with Task Bench build system:

```bash
# In build_all.sh
if [[ $USE_CUDASTF -eq 1 ]]; then
    make -C cudastf clean
    make -C cudastf -j$THREADS
fi
```

### Dependency Management

```bash
# In get_deps.sh
export USE_CUDASTF=${USE_CUDASTF:-$DEFAULT_FEATURES}

if [[ $USE_CUDASTF -eq 1 ]]; then
    export CUDASTF_DIR="$TASKBENCH_DEPS_DIR"/cudastf
    export CCCL_DIR="$CUDASTF_DIR"/cccl
fi
```

## Error Handling and Debugging

### Common STF Patterns

1. **Proper Data Initialization**:
```cpp
// Always initialize host data before STF
std::vector<char> host_data(size, 0);
auto logical_data = ctx.logical_data(host_data.data(), size);
```

2. **Correct Access Patterns**:
```cpp
// Use appropriate access specifiers
ctx.task(data.read()).apply(read_kernel);   // Read-only
ctx.task(data.write()).apply(write_kernel); // Write-only
ctx.task(data.rw()).apply(modify_kernel);   // Read-write
```

3. **Kernel Launch Configuration**:
```cpp
// Proper block/grid sizing
int threads_per_block = 256;
int blocks = (output.size() + threads_per_block - 1) / threads_per_block;
```

### Debug Techniques

1. **STF Debug Output**: Enable STF internal debugging
2. **CUDA Error Checking**: Check kernel launch errors
3. **Memory Validation**: Verify data integrity
4. **Performance Profiling**: Use NVIDIA Nsight tools

### Performance Optimization

1. **Kernel Optimization**: Optimize arithmetic intensity
2. **Memory Coalescing**: Ensure coalesced memory access
3. **Occupancy**: Maximize GPU occupancy
4. **STF Efficiency**: Minimize STF overhead

## Future Enhancements

### Potential Improvements

1. **Multi-GPU Support**: Extend to multiple GPUs
2. **Advanced Kernels**: Implement DGEMM and DAXPY kernels
3. **Memory Optimization**: Advanced memory management
4. **Performance Tuning**: Auto-tuning for different architectures

### Integration Opportunities

1. **NCCL Integration**: Multi-GPU communication
2. **cuBLAS Integration**: Optimized linear algebra
3. **Thrust Integration**: Parallel algorithms
4. **CUB Integration**: Collective primitives

## References

- [CUDA STF Documentation](https://nvidia.github.io/cccl/cudax/api/stream_task_framework.html)
- [CCCL Project](https://github.com/NVIDIA/cccl)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Task Bench Architecture](https://github.com/StanfordLegion/task-bench)
