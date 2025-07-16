#!/bin/bash

# CUDA STF Test Suite for Task Bench
# This script runs comprehensive tests to validate the CUDA STF implementation

set -e  # Exit on any error

echo "=========================================="
echo "CUDA STF Test Suite for Task Bench"
echo "=========================================="

# Check if binary exists
if [ ! -f "./main" ]; then
    echo "Error: ./main binary not found. Please run 'make' first."
    exit 1
fi

# Test counter
TESTS_PASSED=0
TESTS_TOTAL=0

# Function to run a test and check if it succeeds
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -n "Testing $test_name... "
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    if eval "$command" > /dev/null 2>&1; then
        echo "PASS"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo "FAIL"
        echo "  Command: $command"
    fi
}

echo ""
echo "1. Basic Functionality Tests"
echo "----------------------------"

# Basic execution test
run_test "Basic execution" "./main -steps 2 -width 2"

# Help test
run_test "Help display" "./main --help"

echo ""
echo "2. Dependency Pattern Tests"
echo "---------------------------"

# Test all dependency patterns
run_test "Trivial pattern" "./main -type trivial -steps 2 -width 2"
run_test "No-comm pattern" "./main -type no_comm -steps 2 -width 2"
run_test "Stencil 1D pattern" "./main -type stencil_1d -steps 3 -width 3"
run_test "Tree pattern" "./main -type tree -steps 3 -width 4"
run_test "Dom pattern" "./main -type dom -steps 2 -width 2"
run_test "FFT pattern" "./main -type fft -steps 2 -width 4"
run_test "All-to-all pattern" "./main -type all_to_all -steps 2 -width 2"
run_test "Nearest pattern" "./main -type nearest -steps 2 -width 2"
run_test "Spread pattern" "./main -type spread -steps 2 -width 2"
run_test "Random nearest pattern" "./main -type random_nearest -steps 2 -width 2"

echo ""
echo "3. Kernel Type Tests"
echo "--------------------"

# Test all kernel types
run_test "Empty kernel" "./main -kernel empty -steps 2 -width 2"
run_test "Compute-bound kernel" "./main -kernel compute_bound -iter 100 -steps 2 -width 2"
run_test "Memory-bound kernel" "./main -kernel memory_bound -iter 100 -steps 2 -width 2"
run_test "Busy-wait kernel" "./main -kernel busy_wait -iter 100 -steps 2 -width 2"
run_test "Compute-bound2 kernel" "./main -kernel compute_bound2 -iter 100 -steps 2 -width 2"
run_test "Load imbalance kernel" "./main -kernel load_imbalance -iter 100 -imbalance 0.1 -steps 2 -width 2"

# Test fallback kernels
run_test "DGEMM fallback" "./main -kernel compute_dgemm -iter 100 -steps 2 -width 2"
run_test "DAXPY fallback" "./main -kernel memory_daxpy -iter 100 -steps 2 -width 2"
run_test "IO-bound fallback" "./main -kernel io_bound -steps 2 -width 2"

echo ""
echo "4. Parameter Validation Tests"
echo "-----------------------------"

# Test various parameter combinations
run_test "Large iteration count" "./main -kernel compute_bound -iter 5000 -steps 2 -width 2"
run_test "Large scratch size" "./main -kernel memory_bound -scratch 2048 -iter 100 -steps 2 -width 2"
run_test "High imbalance" "./main -kernel load_imbalance -imbalance 0.5 -iter 100 -steps 2 -width 2"
run_test "Large task graph" "./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 500"

echo ""
echo "5. Performance Tests"
echo "--------------------"

# Performance validation tests
run_test "Small scale performance" "./main -type stencil_1d -steps 3 -width 4 -kernel compute_bound -iter 1000"
run_test "Medium scale performance" "./main -type stencil_1d -steps 5 -width 8 -kernel compute_bound -iter 2000"
run_test "Memory performance" "./main -type stencil_1d -steps 3 -width 4 -kernel memory_bound -iter 1000 -scratch 1024"

echo ""
echo "6. Edge Case Tests"
echo "------------------"

# Edge cases
run_test "Single task" "./main -steps 1 -width 1 -kernel compute_bound -iter 100"
run_test "Zero iterations" "./main -kernel compute_bound -iter 0 -steps 2 -width 2"
run_test "Minimal imbalance" "./main -kernel load_imbalance -imbalance 0.0 -iter 100 -steps 2 -width 2"
run_test "Maximum imbalance" "./main -kernel load_imbalance -imbalance 1.0 -iter 100 -steps 2 -width 2"

echo ""
echo "7. Stress Tests"
echo "---------------"

# Stress tests (optional, may take longer)
if [ "${CUDA_STF_STRESS_TEST:-0}" = "1" ]; then
    run_test "Large task count" "./main -type stencil_1d -steps 8 -width 16 -kernel compute_bound -iter 1000"
    run_test "High iteration count" "./main -kernel compute_bound -iter 50000 -steps 2 -width 2"
    run_test "Large memory usage" "./main -kernel memory_bound -scratch 8192 -iter 1000 -steps 3 -width 4"
else
    echo "Skipping stress tests (set CUDA_STF_STRESS_TEST=1 to enable)"
fi

echo ""
echo "=========================================="
echo "Test Results Summary"
echo "=========================================="
echo "Tests passed: $TESTS_PASSED / $TESTS_TOTAL"

if [ $TESTS_PASSED -eq $TESTS_TOTAL ]; then
    echo "Status: ALL TESTS PASSED ✅"
    exit 0
else
    TESTS_FAILED=$((TESTS_TOTAL - TESTS_PASSED))
    echo "Status: $TESTS_FAILED TESTS FAILED ❌"
    exit 1
fi
