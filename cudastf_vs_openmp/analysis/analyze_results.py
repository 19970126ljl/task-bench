#!/usr/bin/env python3

"""
Comprehensive analysis and visualization script for CUDASTF vs OpenMP benchmarks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(results_file):
    """Load and clean benchmark results"""
    try:
        df = pd.read_csv(results_file)
        
        # Clean data
        df = df[df['status'] == 'SUCCESS'].copy()
        
        # Convert numeric columns
        numeric_cols = ['wall_time', 'total_time', 'task_throughput']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['wall_time'])
        
        print(f"Loaded {len(df)} successful benchmark results")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def generate_performance_comparison(df, output_dir):
    """Generate performance comparison plots"""
    
    # 1. Overall performance comparison by task type
    plt.figure(figsize=(15, 10))
    
    # Group by implementation and task type, calculate mean wall time
    perf_summary = df.groupby(['implementation', 'task_type'])['wall_time'].mean().reset_index()
    perf_pivot = perf_summary.pivot(index='task_type', columns='implementation', values='wall_time')
    
    # Calculate speedup (OpenMP baseline)
    if 'openmp' in perf_pivot.columns and 'cudastf' in perf_pivot.columns:
        perf_pivot['speedup'] = perf_pivot['openmp'] / perf_pivot['cudastf']
    
    # Plot comparison
    ax = perf_pivot[['openmp', 'cudastf']].plot(kind='bar', figsize=(15, 8))
    plt.title('Performance Comparison: CUDASTF vs OpenMP (Wall Time)', fontsize=16)
    plt.xlabel('Task Type', fontsize=12)
    plt.ylabel('Wall Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(['OpenMP', 'CUDASTF'])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison_by_task.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Speedup analysis
    if 'speedup' in perf_pivot.columns:
        plt.figure(figsize=(12, 8))
        speedup_data = perf_pivot['speedup'].dropna()
        bars = plt.bar(range(len(speedup_data)), speedup_data.values)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No speedup')
        plt.title('CUDASTF Speedup over OpenMP', fontsize=16)
        plt.xlabel('Task Type', fontsize=12)
        plt.ylabel('Speedup Factor', fontsize=12)
        plt.xticks(range(len(speedup_data)), speedup_data.index, rotation=45)
        
        # Color bars based on speedup
        for i, bar in enumerate(bars):
            if speedup_data.iloc[i] > 1.0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/speedup_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_kernel_analysis(df, output_dir):
    """Analyze performance by kernel type"""
    
    plt.figure(figsize=(15, 10))
    
    # Performance by kernel type
    kernel_perf = df.groupby(['implementation', 'kernel'])['wall_time'].mean().reset_index()
    kernel_pivot = kernel_perf.pivot(index='kernel', columns='implementation', values='wall_time')
    
    ax = kernel_pivot.plot(kind='bar', figsize=(15, 8))
    plt.title('Performance by Kernel Type', fontsize=16)
    plt.xlabel('Kernel Type', fontsize=12)
    plt.ylabel('Wall Time (seconds)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_by_kernel.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_scaling_analysis(df, output_dir):
    """Analyze scaling with problem size (steps)"""
    
    plt.figure(figsize=(15, 10))
    
    # Scaling analysis
    scaling_data = df.groupby(['implementation', 'steps'])['wall_time'].mean().reset_index()
    
    for impl in scaling_data['implementation'].unique():
        impl_data = scaling_data[scaling_data['implementation'] == impl]
        plt.plot(impl_data['steps'], impl_data['wall_time'], 
                marker='o', linewidth=2, label=impl.upper())
    
    plt.title('Scaling Analysis: Performance vs Problem Size', fontsize=16)
    plt.xlabel('Number of Steps', fontsize=12)
    plt.ylabel('Wall Time (seconds)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/scaling_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_heatmap_analysis(df, output_dir):
    """Generate heatmap of performance across task types and kernels"""
    
    # Create heatmap data for each implementation
    for impl in df['implementation'].unique():
        impl_data = df[df['implementation'] == impl]
        heatmap_data = impl_data.groupby(['task_type', 'kernel'])['wall_time'].mean().unstack()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd')
        plt.title(f'{impl.upper()} Performance Heatmap (Wall Time)', fontsize=16)
        plt.xlabel('Kernel Type', fontsize=12)
        plt.ylabel('Task Type', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_{impl}.png", dpi=300, bbox_inches='tight')
        plt.close()

def generate_statistical_summary(df, output_dir):
    """Generate detailed statistical summary"""
    
    # Overall statistics
    stats_summary = df.groupby(['implementation']).agg({
        'wall_time': ['count', 'mean', 'std', 'min', 'max', 'median'],
        'total_time': ['mean', 'std'],
        'task_throughput': ['mean', 'std']
    }).round(6)
    
    stats_summary.to_csv(f"{output_dir}/statistical_summary.csv")
    
    # Detailed breakdown
    detailed_stats = df.groupby(['implementation', 'task_type', 'kernel']).agg({
        'wall_time': ['mean', 'std', 'min', 'max'],
        'total_time': 'mean',
        'task_throughput': 'mean'
    }).round(6)
    
    detailed_stats.to_csv(f"{output_dir}/detailed_statistics.csv")
    
    print("Statistical summaries saved to CSV files")

def generate_performance_report(df, output_dir):
    """Generate a comprehensive performance report"""
    
    report_file = f"{output_dir}/performance_report.md"
    
    with open(report_file, 'w') as f:
        f.write("# CUDASTF vs OpenMP Performance Analysis Report\n\n")
        f.write(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        # Overall summary
        f.write("## Executive Summary\n\n")
        
        total_tests = len(df)
        implementations = df['implementation'].unique()
        
        f.write(f"- Total successful benchmark runs: {total_tests}\n")
        f.write(f"- Implementations tested: {', '.join(implementations)}\n")
        f.write(f"- Task types tested: {len(df['task_type'].unique())}\n")
        f.write(f"- Kernel variants tested: {len(df['kernel'].unique())}\n\n")
        
        # Performance comparison
        if 'openmp' in implementations and 'cudastf' in implementations:
            openmp_mean = df[df['implementation'] == 'openmp']['wall_time'].mean()
            cudastf_mean = df[df['implementation'] == 'cudastf']['wall_time'].mean()
            overall_speedup = openmp_mean / cudastf_mean
            
            f.write("## Overall Performance\n\n")
            f.write(f"- OpenMP average wall time: {openmp_mean:.6f} seconds\n")
            f.write(f"- CUDASTF average wall time: {cudastf_mean:.6f} seconds\n")
            f.write(f"- Overall speedup factor: {overall_speedup:.2f}x\n\n")
            
            if overall_speedup > 1.0:
                f.write("✅ CUDASTF shows overall performance improvement\n\n")
            else:
                f.write("❌ CUDASTF shows performance regression\n\n")
        
        # Best and worst performing scenarios
        f.write("## Performance Analysis by Scenario\n\n")
        
        scenario_perf = df.groupby(['task_type', 'kernel', 'implementation'])['wall_time'].mean().reset_index()
        
        if 'openmp' in implementations and 'cudastf' in implementations:
            # Calculate speedups for each scenario
            openmp_perf = scenario_perf[scenario_perf['implementation'] == 'openmp'].set_index(['task_type', 'kernel'])
            cudastf_perf = scenario_perf[scenario_perf['implementation'] == 'cudastf'].set_index(['task_type', 'kernel'])
            
            speedups = openmp_perf['wall_time'] / cudastf_perf['wall_time']
            speedups = speedups.dropna().sort_values(ascending=False)
            
            f.write("### Best CUDASTF Performance (Top 5 Speedups)\n\n")
            for i, (scenario, speedup) in enumerate(speedups.head().items()):
                task_type, kernel = scenario
                f.write(f"{i+1}. {task_type} + {kernel}: {speedup:.2f}x speedup\n")
            
            f.write("\n### Worst CUDASTF Performance (Bottom 5 Speedups)\n\n")
            for i, (scenario, speedup) in enumerate(speedups.tail().items()):
                task_type, kernel = scenario
                f.write(f"{i+1}. {task_type} + {kernel}: {speedup:.2f}x speedup\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on the benchmark results, consider the following improvements:\n\n")
        f.write("1. **Performance Bottlenecks**: Focus on scenarios with speedup < 1.0\n")
        f.write("2. **Memory Management**: Analyze memory-bound kernel performance\n")
        f.write("3. **Task Scheduling**: Optimize task graph construction and execution\n")
        f.write("4. **GPU Utilization**: Ensure efficient GPU resource utilization\n")
        f.write("5. **Communication Patterns**: Optimize data transfer patterns\n\n")
    
    print(f"Performance report saved to: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze CUDASTF vs OpenMP benchmark results')
    parser.add_argument('results_file', help='Path to benchmark results CSV file')
    parser.add_argument('--output-dir', default='./analysis_output', 
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and analyze data
    df = load_and_clean_data(args.results_file)
    
    print("Generating performance comparison plots...")
    generate_performance_comparison(df, args.output_dir)
    
    print("Generating kernel analysis...")
    generate_kernel_analysis(df, args.output_dir)
    
    print("Generating scaling analysis...")
    generate_scaling_analysis(df, args.output_dir)
    
    print("Generating heatmap analysis...")
    generate_heatmap_analysis(df, args.output_dir)
    
    print("Generating statistical summary...")
    generate_statistical_summary(df, args.output_dir)
    
    print("Generating performance report...")
    generate_performance_report(df, args.output_dir)
    
    print(f"\nAnalysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
