#!/bin/bash

# Comprehensive benchmark script for CUDASTF vs OpenMP comparison
# Collects detailed performance data for analysis

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="../results/benchmark_${TIMESTAMP}"
WARMUP_RUNS=2
MEASUREMENT_RUNS=5
TASK_WIDTH=4
RUN_OPENMP=1
RUN_CUDASTF=1
QUICK_MODE=0

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -w, --width N         Task width (default: 4)"
    echo "  -r, --runs N          Measurement runs per test (default: 5)"
    echo "  -u, --warmup N        Warmup runs per test (default: 2)"
    echo "  --openmp-only         Run only OpenMP benchmarks"
    echo "  --cudastf-only        Run only CUDASTF benchmarks"
    echo "  --quick               Quick mode: fewer task types and steps"
    echo "  -h, --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run full benchmark with default settings"
    echo "  $0 -w 8 -r 3          # Use width=8, 3 measurement runs"
    echo "  $0 --quick            # Quick benchmark for testing"
    echo "  $0 --openmp-only      # Only test OpenMP implementation"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--width)
            TASK_WIDTH="$2"
            shift 2
            ;;
        -r|--runs)
            MEASUREMENT_RUNS="$2"
            shift 2
            ;;
        -u|--warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --openmp-only)
            RUN_OPENMP=1
            RUN_CUDASTF=0
            shift
            ;;
        --cudastf-only)
            RUN_OPENMP=0
            RUN_CUDASTF=1
            shift
            ;;
        --quick)
            QUICK_MODE=1
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

echo -e "${GREEN}=== Comprehensive CUDASTF vs OpenMP Benchmark ===${NC}"
echo "Configuration:"
echo "  Task width: $TASK_WIDTH"
echo "  Measurement runs: $MEASUREMENT_RUNS"
echo "  Warmup runs: $WARMUP_RUNS"
echo "  Run OpenMP: $([[ $RUN_OPENMP -eq 1 ]] && echo "Yes" || echo "No")"
echo "  Run CUDASTF: $([[ $RUN_CUDASTF -eq 1 ]] && echo "Yes" || echo "No")"
echo "  Quick mode: $([[ $QUICK_MODE -eq 1 ]] && echo "Yes" || echo "No")"
echo "Results will be saved to: $RESULTS_DIR"

# Create results directory
mkdir -p "$RESULTS_DIR"

# Check dependencies
if [[ ! -d ../../deps ]]; then
    echo -e "${RED}Error: deps directory not found. Run ./get_deps.sh first.${NC}"
    exit 1
fi

source ../../deps/env.sh

# Configure test parameters based on mode
if [[ $QUICK_MODE -eq 1 ]]; then
    # Quick mode: fewer configurations for testing
    declare -a task_types=(
        "trivial"
        "no_comm"
        "stencil_1d"
        "tree"
    )
    declare -a step_sizes=(5 10 15)
else
    # Full mode: comprehensive testing
    declare -a task_types=(
        "trivial"
        "no_comm"
        "stencil_1d"
        "stencil_1d_periodic"
        "dom"
        "tree"
        "fft"
        "nearest"
        "spread -period 2"
        "random_nearest"
    )
    declare -a step_sizes=(5 10 15 20 25 30)
fi

declare -a kernel_configs=(
    ""
    "-kernel compute_bound -iter 1024"
    "-kernel memory_bound -iter 1024 -scratch $((64*16))"
    "-kernel load_imbalance -iter 1024 -imbalance 0.1"
    "-output 1024"
)

declare -a kernel_names=(
    "default"
    "compute_bound"
    "memory_bound"
    "load_imbalance"
    "communication_bound"
)

# Function to run single benchmark
run_single_benchmark() {
    local impl=$1
    local binary=$2
    local worker_flag=$3
    local task_type=$4
    local kernel_config=$5
    local kernel_name=$6
    local steps=$7
    local run_id=$8
    
    local cmd="../../$impl/$binary -steps $steps -width $TASK_WIDTH -type $task_type $kernel_config"
    if [[ -n "$worker_flag" ]]; then
        cmd="$cmd $worker_flag"
    fi
    
    echo "Running: $impl - $task_type - $kernel_name - steps:$steps - run:$run_id"
    
    # Capture timing and output
    local start_time=$(date +%s.%N)
    local output_file="$RESULTS_DIR/${impl}_${task_type}_${kernel_name}_steps${steps}_run${run_id}.out"
    local error_file="$RESULTS_DIR/${impl}_${task_type}_${kernel_name}_steps${steps}_run${run_id}.err"
    
    if timeout 300 bash -c "$cmd" > "$output_file" 2> "$error_file"; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        
        # Extract performance metrics from output
        local total_time=$(grep "Total time:" "$output_file" | awk '{print $3}' || echo "N/A")
        local task_throughput=$(grep "Task throughput:" "$output_file" | awk '{print $3}' || echo "N/A")
        
        # Log to CSV
        echo "$impl,$task_type,$kernel_name,$steps,$run_id,$duration,$total_time,$task_throughput,SUCCESS" >> "$RESULTS_DIR/benchmark_results.csv"
        
        return 0
    else
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        echo "$impl,$task_type,$kernel_name,$steps,$run_id,$duration,N/A,N/A,FAILED" >> "$RESULTS_DIR/benchmark_results.csv"
        echo -e "${RED}FAILED: $cmd${NC}"
        return 1
    fi
}

# Function to run implementation benchmarks
run_implementation_benchmarks() {
    local impl=$1
    local binary=$2
    local worker_flag=$3
    
    echo -e "${YELLOW}Running $impl benchmarks...${NC}"
    
    local total_tests=0
    local passed_tests=0
    
    for task_type in "${task_types[@]}"; do
        for i in "${!kernel_configs[@]}"; do
            local kernel_config="${kernel_configs[$i]}"
            local kernel_name="${kernel_names[$i]}"
            
            for steps in "${step_sizes[@]}"; do
                # Warmup runs
                for warmup in $(seq 1 $WARMUP_RUNS); do
                    echo "  Warmup $warmup/$WARMUP_RUNS"
                    run_single_benchmark "$impl" "$binary" "$worker_flag" "$task_type" "$kernel_config" "$kernel_name" "$steps" "warmup$warmup" > /dev/null 2>&1 || true
                done
                
                # Measurement runs
                for run in $(seq 1 $MEASUREMENT_RUNS); do
                    total_tests=$((total_tests + 1))
                    if run_single_benchmark "$impl" "$binary" "$worker_flag" "$task_type" "$kernel_config" "$kernel_name" "$steps" "$run"; then
                        passed_tests=$((passed_tests + 1))
                    fi
                done
            done
        done
    done
    
    echo -e "${GREEN}$impl: $passed_tests/$total_tests tests passed${NC}"
    return $((total_tests - passed_tests))
}

# Initialize CSV file
echo "implementation,task_type,kernel,steps,run_id,wall_time,total_time,task_throughput,status" > "$RESULTS_DIR/benchmark_results.csv"

# Build implementations
echo -e "${YELLOW}Building implementations...${NC}"

echo "Building OpenMP..."
cd ../../openmp
make clean > /dev/null 2>&1
if ! make -j$(nproc) > /dev/null 2>&1; then
    echo -e "${RED}Failed to build OpenMP${NC}"
    exit 1
fi
cd ../cudastf_vs_openmp/scripts

echo "Building CUDASTF..."
cd ../../cudastf
make clean > /dev/null 2>&1
if ! make -j$(nproc) > /dev/null 2>&1; then
    echo -e "${RED}Failed to build CUDASTF${NC}"
    exit 1
fi
cd ../cudastf_vs_openmp/scripts

# Run benchmarks
total_failed=0

# OpenMP benchmarks
if [[ $RUN_OPENMP -eq 1 ]] && [[ "$USE_OPENMP" == "1" ]]; then
    export LD_LIBRARY_PATH=/usr/local/clang/lib:$LD_LIBRARY_PATH
    run_implementation_benchmarks "openmp" "main" "-worker 2"
    total_failed=$((total_failed + $?))
elif [[ $RUN_OPENMP -eq 1 ]]; then
    echo -e "${YELLOW}OpenMP benchmarks skipped (USE_OPENMP != 1)${NC}"
else
    echo -e "${YELLOW}OpenMP benchmarks skipped (--cudastf-only specified)${NC}"
fi

# CUDASTF benchmarks
if [[ $RUN_CUDASTF -eq 1 ]]; then
    run_implementation_benchmarks "cudastf" "main" ""
    total_failed=$((total_failed + $?))
else
    echo -e "${YELLOW}CUDASTF benchmarks skipped (--openmp-only specified)${NC}"
fi

# Generate summary
echo -e "${BLUE}Generating benchmark summary...${NC}"
python3 - << EOF
import pandas as pd
import sys

try:
    df = pd.read_csv('$RESULTS_DIR/benchmark_results.csv')
    
    # Summary statistics
    summary = df.groupby(['implementation', 'task_type', 'kernel']).agg({
        'wall_time': ['mean', 'std', 'min', 'max'],
        'total_time': lambda x: x.replace('N/A', None).astype(float).mean(),
        'task_throughput': lambda x: x.replace('N/A', None).astype(float).mean(),
        'status': lambda x: (x == 'SUCCESS').sum()
    }).round(6)
    
    summary.to_csv('$RESULTS_DIR/benchmark_summary.csv')
    print("Summary saved to benchmark_summary.csv")
    
except Exception as e:
    print(f"Error generating summary: {e}")
    sys.exit(1)
EOF

echo
if [[ $total_failed -eq 0 ]]; then
    echo -e "${GREEN}All benchmarks completed successfully!${NC}"
    echo "Results saved to: $RESULTS_DIR"
else
    echo -e "${RED}$total_failed tests failed${NC}"
    echo "Results saved to: $RESULTS_DIR"
    exit 1
fi
