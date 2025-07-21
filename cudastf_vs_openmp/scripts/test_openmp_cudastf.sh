#!/bin/bash

# Test script for OpenMP and CUDA STF implementations
# Based on test_all.sh but focused only on OpenMP and CUDA STF

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== OpenMP and CUDA STF Test Script ===${NC}"

# Check if dependencies exist
if [[ ! -d deps ]]; then
    echo -e "${RED}Error: The directory deps does not exist.${NC}"
    echo "Please run ./get_deps.sh and try again."
    exit 1
fi

source deps/env.sh

# Test configurations
basic_types=(
    trivial
    no_comm
    stencil_1d
    stencil_1d_periodic
    dom
    tree
    fft
    nearest
    "spread -period 2"
    random_nearest
)

compute_bound="-kernel compute_bound -iter 1024"
memory_bound="-kernel memory_bound -iter 1024 -scratch $((64*16))"
imbalanced="-kernel load_imbalance -iter 1024 -imbalance 0.1"
communication_bound="-output 1024"

kernels=("" "$compute_bound" "$memory_bound" "$imbalanced" "$communication_bound")
steps=23 # chosen to be relatively prime with 2, 3, 5

# Function to run tests
run_tests() {
    local impl=$1
    local binary=$2
    local worker_flag=$3
    
    echo -e "${YELLOW}Running $impl tests...${NC}"
    
    local count=0
    local passed=0
    
    for t in "${basic_types[@]}"; do
        for k in "${kernels[@]}"; do
            count=$((count + 1))
            
            # Build command
            local cmd="./$impl/$binary -steps $steps -type $t $k"
            if [[ -n "$worker_flag" ]]; then
                cmd="$cmd $worker_flag"
            fi
            
            echo "  Test $count: $cmd"
            if eval "$cmd"; then
                passed=$((passed + 1))
            else
                echo -e "${RED}    FAILED: $cmd${NC}"
            fi
            
            # Test with multiple graphs
            local cmd2="./$impl/$binary -steps $steps -type $t $k -and -steps $steps -type $t $k"
            if [[ -n "$worker_flag" ]]; then
                cmd2="$cmd2 $worker_flag"
            fi
            
            count=$((count + 1))
            echo "  Test $count: $cmd2"
            if eval "$cmd2"; then
                passed=$((passed + 1))
            else
                echo -e "${RED}    FAILED: $cmd2${NC}"
            fi
        done
    done
    
    echo -e "${GREEN}$impl: $passed/$count tests passed${NC}"
    return $((count - passed))
}

# Build phase
echo -e "${YELLOW}Building OpenMP...${NC}"
cd openmp
make clean > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1 || {
    echo -e "${RED}Failed to build OpenMP${NC}"
    exit 1
}
cd ..

echo -e "${YELLOW}Building CUDA STF...${NC}"
cd cudastf
make clean > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1 || {
    echo -e "${RED}Failed to build CUDA STF${NC}"
    exit 1
}
cd ..

# Test phase
total_failed=0

# Test OpenMP
if [[ "$USE_OPENMP" == "1" ]]; then
    export LD_LIBRARY_PATH=/usr/local/clang/lib:$LD_LIBRARY_PATH
    run_tests "openmp" "main" "-worker 2"
    total_failed=$((total_failed + $?))
else
    echo -e "${YELLOW}OpenMP tests skipped (USE_OPENMP != 1)${NC}"
fi

# Test CUDA STF
run_tests "cudastf" "main" ""
total_failed=$((total_failed + $?))

# Summary
echo
if [[ $total_failed -eq 0 ]]; then
    echo -e "${GREEN}All OpenMP and CUDA STF tests completed successfully!${NC}"
else
    echo -e "${RED}$total_failed tests failed${NC}"
    exit 1
fi