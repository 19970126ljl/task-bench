# OpenMP and CUDA STF Testing Guide

This directory contains specialized test scripts for running OpenMP and CUDA STF implementations of Task Bench.

## Test Scripts

### 1. `test_openmp_cudastf.sh`
**Full test suite** for OpenMP and CUDA STF implementations.

**Features:**
- Comprehensive testing with all basic task types
- Multiple kernel variants (compute-bound, memory-bound, imbalanced, communication-bound)
- Parallel graph execution testing
- Detailed progress reporting with colored output
- Error handling and summary statistics

**Usage:**
```bash
./test_openmp_cudastf.sh
```

**Environment Variables:**
- `USE_OPENMP=1` - Enable OpenMP tests (default: enabled if built)
- Requires dependencies from `./get_deps.sh`

### 2. `test_openmp_cudastf_quick.sh`
**Quick validation** script for faster testing.

**Features:**
- Reduced test set with 3 basic task types
- Smaller step count for faster execution
- Essential kernel variants only
- Quick build and test cycle

**Usage:**
```bash
./test_openmp_cudastf_quick.sh
```

## Task Types Tested

Both scripts test the following task types:
- `trivial` - Simple independent tasks
- `no_comm` - Tasks with no communication
- `stencil_1d` - 1D stencil communication pattern
- `stencil_1d_periodic` - 1D stencil with periodic boundaries
- `dom` - Domain decomposition pattern
- `tree` - Tree-based communication
- `fft` - FFT communication pattern
- `nearest` - Nearest neighbor communication
- `spread -period 2` - Spread communication with period 2
- `random_nearest` - Random nearest neighbor pattern

## Kernel Variants

- **Default** - Basic task execution
- **Compute-bound** - CPU-intensive tasks with 1024 iterations
- **Memory-bound** - Memory-intensive tasks with 1024 iterations and 1KB scratch
- **Imbalanced** - Load imbalanced tasks with 10% imbalance
- **Communication-bound** - Communication-heavy tasks with 1KB output

## Build Requirements

### OpenMP
- GCC with OpenMP support (`-fopenmp`)
- Standard C++11 compiler

### CUDA STF
- NVIDIA CUDA compiler (`nvcc`)
- CUDA-capable GPU
- CCCL (CUDA C++ Core Libraries) - automatically configured via `deps/env.sh`

## Directory Structure

```
task-bench/
├── openmp/           # OpenMP implementation
│   ├── main.cc       # Main OpenMP benchmark
│   ├── Makefile      # Build configuration
│   └── ...
├── cudastf/          # CUDA STF implementation
│   ├── main.cu       # Main CUDA STF benchmark
│   ├── Makefile      # Build configuration
│   └── ...
├── test_openmp_cudastf.sh      # Full test suite
├── test_openmp_cudastf_quick.sh # Quick test suite
└── README_OPENMP_CUDASTF.md    # This file
```

## Example Output

```
=== OpenMP and CUDA STF Test Script ===
Building OpenMP...
Building CUDA STF...
Running OpenMP tests...
  Test 1: ./openmp/main -steps 23 -type trivial -worker 2
  Test 2: ./openmp/main -steps 23 -type trivial -and -steps 23 -type trivial -worker 2
  ...
OpenMP: 80/80 tests passed
Running CUDA STF tests...
  Test 1: ./cudastf/main -steps 23 -type trivial
  Test 2: ./cudastf/main -steps 23 -type trivial -and -steps 23 -type trivial
  ...
CUDA STF: 80/80 tests passed
All OpenMP and CUDA STF tests completed successfully!
```

## Troubleshooting

### Common Issues

1. **Build failures**
   - Ensure dependencies are installed: `./get_deps.sh`
   - Check CUDA installation for cudastf
   - Verify OpenMP support in compiler

2. **Runtime errors**
   - Check GPU availability for CUDA STF: `nvidia-smi`
   - Verify OpenMP thread support: `echo $OMP_NUM_THREADS`

3. **Missing libraries**
   - Source environment: `source deps/env.sh`
   - Check LD_LIBRARY_PATH for OpenMP runtime

### Debug Mode

To enable debug builds:
```bash
cd openmp && make DEBUG=1 clean && make DEBUG=1
cd cudastf && make DEBUG=1 clean && make DEBUG=1