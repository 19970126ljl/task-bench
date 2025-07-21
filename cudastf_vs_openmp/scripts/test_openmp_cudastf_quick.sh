#!/bin/bash

# Quick test script for OpenMP and CUDA STF
# Runs a subset of tests for faster validation

set -e

if [[ ! -d deps ]]; then
    echo "Error: deps directory not found. Run ./get_deps.sh first."
    exit 1
fi

source deps/env.sh

# Quick test configurations
quick_types=(
    trivial
    no_comm
    stencil_1d
)

compute_bound="-kernel compute_bound -iter 1024"
memory_bound="-kernel memory_bound -iter 1024 -scratch $((64*16))"

kernels=("" "$compute_bound" "$memory_bound")
steps=5  # Smaller steps for quick testing

echo "=== Quick OpenMP and CUDA STF Tests ==="

# Build
# echo "Building OpenMP..."
# cd openmp && make clean && make -j$(nproc) && cd ..

# echo "Building CUDA STF..."
# cd cudastf && make clean && make -j$(nproc) && cd ..

# Run quick tests
echo "Running quick tests..."

# OpenMP tests
if [[ "$USE_OPENMP" == "1" ]]; then
    export LD_LIBRARY_PATH=/usr/local/clang/lib:$LD_LIBRARY_PATH
    for t in "${quick_types[@]}"; do
        for k in "${kernels[@]}"; do
            echo "OpenMP: $t $k"
            ./openmp/main -steps $steps -type $t $k -worker 2
        done
    done
fi

# CUDA STF tests
for t in "${quick_types[@]}"; do
    for k in "${kernels[@]}"; do
        echo "CUDA STF: $t $k"
        ./cudastf/main -steps $steps -type $t $k
    done
done

echo "Quick tests completed successfully!"
