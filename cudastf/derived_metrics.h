#ifndef TASK_BENCH_CUDASTF_DERIVED_METRICS_H
#define TASK_BENCH_CUDASTF_DERIVED_METRICS_H

#include <cstdint>
#include <string>
#include <vector>

#include "arguments.h"
#include "expanded_dag.h"
#include "task_profile.h"

struct SampleResult;

struct MeasuredTaskDuration {
  DagTaskKey key;
  int configured_device = 0;
  double measured_duration_ms = 0.0;
};

struct DurationStatistics {
  double mean_ms = 0.0;
  double median_ms = 0.0;
  double p95_ms = 0.0;
  double cv = 0.0;
};

struct ParallelismMetrics {
  double work_ms = 0.0;
  double critical_path_ms = 0.0;
  double average = 0.0;
  std::uint64_t peak = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double cv = 0.0;
};

struct DagParallelismMetrics {
  std::int64_t dag_index = 0;
  std::vector<MeasuredTaskDuration> tasks;
  DurationStatistics task_duration;
  ParallelismMetrics parallelism;
};

struct ParallelismAnalysis {
  bool available = false;
  std::string unavailable_reason;
  std::vector<DagParallelismMetrics> dags;
  DurationStatistics combined_task_duration;
  ParallelismMetrics combined_parallelism;
};

struct DerivedMetrics {
  ParallelismAnalysis parallelism;
};

DerivedMetrics derive_metrics(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples);

#endif
