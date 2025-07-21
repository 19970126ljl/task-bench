# CUDASTF vs OpenMP Performance Analysis

This directory contains a comprehensive benchmarking and analysis framework for comparing CUDASTF and OpenMP implementations of Task Bench.

## Quick Start

### Basic Usage

```bash
# Quick validation test
./quick_test.sh

# Full analysis with default settings
./run_complete_analysis.sh

# Quick benchmark for testing
./run_complete_analysis.sh --quick

# Custom configuration
./run_complete_analysis.sh -w 8 -r 3 --quick
```

### Configuration Options

The analysis script supports various configuration options:

**Benchmark Configuration:**
- `-w, --width N` - Task width (default: 4)
- `-r, --runs N` - Measurement runs per test (default: 5)
- `-u, --warmup N` - Warmup runs per test (default: 2)
- `--quick` - Quick mode with fewer task types and steps

**Implementation Selection:**
- `--openmp-only` - Run only OpenMP benchmarks
- `--cudastf-only` - Run only CUDASTF benchmarks

**Execution Control:**
- `--skip-build` - Skip building implementations
- `--analysis-only DIR` - Skip benchmarks, analyze existing results

### Examples

```bash
# Test only CUDASTF with custom width
./run_complete_analysis.sh --cudastf-only -w 8

# Quick comparison test
./run_complete_analysis.sh --quick -w 4 -r 3

# Analyze existing results without running new benchmarks
./run_complete_analysis.sh --analysis-only results/benchmark_20240101_120000

# Full benchmark with larger task width
./run_complete_analysis.sh -w 16 -r 5
```

The complete analysis will:
1. Build both implementations (unless --skip-build)
2. Run comprehensive benchmarks with specified configuration
3. Analyze results and generate visualizations
4. Create detailed performance reports

## Directory Structure

```
cudastf_vs_openmp/
├── scripts/                          # Benchmark scripts
│   ├── comprehensive_benchmark.sh     # Main benchmark script
│   ├── test_openmp_cudastf.sh        # Original test script
│   └── test_openmp_cudastf_quick.sh   # Quick validation script
├── analysis/                         # Analysis tools
│   └── analyze_results.py            # Data analysis and visualization
├── results/                          # Benchmark results (auto-generated)
│   └── benchmark_YYYYMMDD_HHMMSS/    # Timestamped result directories
├── docs/                             # Documentation
│   └── README_OPENMP_CUDASTF.md      # Original documentation
└── README.md                         # This file
```

## Benchmark Configuration

### Task Types Tested
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

### Kernel Variants
- **Default** - Basic task execution
- **Compute-bound** - CPU-intensive tasks (1024 iterations)
- **Memory-bound** - Memory-intensive tasks (1024 iterations, 1KB scratch)
- **Load-imbalance** - Imbalanced tasks (10% imbalance)
- **Communication-bound** - Communication-heavy tasks (1KB output)

### Problem Sizes
- **Full mode**: Step counts: 5, 10, 15, 20, 25, 30
- **Quick mode**: Step counts: 5, 10, 15
- **Task width**: Configurable (default: 4)
- Multiple runs for statistical significance (configurable: default 5 measurement + 2 warmup runs)

## Individual Scripts

### 1. Comprehensive Benchmark (`scripts/comprehensive_benchmark.sh`)

Runs detailed performance benchmarks with:
- Multiple runs for statistical significance
- Warmup runs to eliminate cold-start effects
- Timeout protection (5 minutes per test)
- CSV output for analysis
- Error handling and logging

**Usage:**
```bash
cd scripts
./comprehensive_benchmark.sh
```

### 2. Analysis Script (`analysis/analyze_results.py`)

Generates comprehensive analysis including:
- Performance comparison plots
- Speedup analysis
- Kernel-specific performance breakdown
- Scaling analysis
- Statistical summaries
- Performance heatmaps

**Usage:**
```bash
cd analysis
python3 analyze_results.py ../results/benchmark_YYYYMMDD_HHMMSS/benchmark_results.csv
```

## Output Files

### Raw Data
- `benchmark_results.csv` - Raw benchmark data
- `benchmark_summary.csv` - Statistical summary

### Analysis Reports
- `performance_report.md` - Comprehensive performance analysis
- `statistical_summary.csv` - Overall statistics
- `detailed_statistics.csv` - Detailed breakdown by scenario

### Visualizations
- `performance_comparison_by_task.png` - Task type comparison
- `speedup_analysis.png` - CUDASTF speedup over OpenMP
- `performance_by_kernel.png` - Kernel type analysis
- `scaling_analysis.png` - Performance vs problem size
- `heatmap_*.png` - Performance heatmaps for each implementation

## Requirements

### System Requirements
- NVIDIA GPU with CUDA support
- GCC with OpenMP support
- Python 3.6+ with packages: pandas, matplotlib, seaborn, numpy
- bc calculator utility

### Environment Setup
```bash
# Install Python dependencies
pip install pandas matplotlib seaborn numpy

# Install bc calculator (Ubuntu/Debian)
sudo apt-get install bc

# Ensure CUDA environment is set up
source ../deps/env.sh
```

## Interpreting Results

### Performance Metrics
- **Wall Time**: Total execution time including overhead
- **Total Time**: Task execution time reported by benchmark
- **Task Throughput**: Tasks completed per second
- **Speedup**: OpenMP time / CUDASTF time (>1.0 means CUDASTF is faster)

### Key Analysis Areas
1. **Overall Performance**: Compare average execution times
2. **Task Type Sensitivity**: Identify which patterns favor each implementation
3. **Kernel Performance**: Understand compute vs memory vs communication bottlenecks
4. **Scaling Behavior**: Analyze performance with increasing problem size
5. **Consistency**: Check standard deviation and outliers

### Performance Optimization Insights
- **Speedup > 1.5**: CUDASTF shows significant advantage
- **Speedup 0.8-1.2**: Performance is comparable
- **Speedup < 0.8**: CUDASTF needs optimization for this scenario

## Troubleshooting

### Common Issues
1. **Build Failures**: Ensure CUDA and OpenMP environments are properly set up
2. **GPU Not Available**: Check `nvidia-smi` output
3. **Python Package Errors**: Install required packages with pip
4. **Permission Errors**: Ensure scripts are executable (`chmod +x`)

### Debug Mode
To run with debug output:
```bash
# Enable verbose output
export DEBUG=1
./run_complete_analysis.sh
```

## Customization

### Modifying Benchmark Parameters
Edit `scripts/comprehensive_benchmark.sh`:
- `step_sizes`: Array of problem sizes to test
- `MEASUREMENT_RUNS`: Number of measurement runs per configuration
- `WARMUP_RUNS`: Number of warmup runs

### Adding New Analysis
Extend `analysis/analyze_results.py`:
- Add new visualization functions
- Implement custom performance metrics
- Create additional statistical analyses

## Contributing

When adding new features:
1. Maintain compatibility with existing data formats
2. Add appropriate error handling
3. Update documentation
4. Test with various configurations

## License

This analysis framework follows the same license as the parent Task Bench project.
