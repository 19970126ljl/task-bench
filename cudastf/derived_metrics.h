#ifndef TASK_BENCH_CUDASTF_DERIVED_METRICS_H
#define TASK_BENCH_CUDASTF_DERIVED_METRICS_H

#include <cstddef>
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
  double task_work_ms = 0.0;
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

struct CombinedParallelismMetrics {
  DurationStatistics task_duration;
  ParallelismMetrics parallelism;
};

struct ParallelismAnalysis {
  bool available = false;
  std::string unavailable_reason;
  std::vector<DagParallelismMetrics> dags;
  CombinedParallelismMetrics combined;
};

struct ConcurrencyMetrics {
  double duration_ms = 0.0;
  double average = 0.0;
  std::uint64_t peak = 0;
  double p50 = 0.0;
  double p95 = 0.0;
  double cv = 0.0;
};

struct EndToEndSpanMetrics {
  bool available = false;
  std::string unavailable_reason;
  ConcurrencyMetrics metrics;
};

struct DagConcurrencyMetrics {
  std::int64_t dag_index = 0;
  double task_work_ms = 0.0;
  DurationStatistics task_duration;
  ConcurrencyMetrics task_gpu_span;
};

struct CombinedConcurrencyMetrics {
  double task_work_ms = 0.0;
  DurationStatistics task_duration;
  ConcurrencyMetrics task_gpu_span;
  EndToEndSpanMetrics end_to_end_span;
};

struct ConcurrencySampleMetrics {
  std::size_t sample_index = 0;
  std::vector<DagConcurrencyMetrics> dags;
  CombinedConcurrencyMetrics combined;
};

struct ScalarSummary {
  double median = 0.0;
  double p95 = 0.0;
};

struct DurationStatisticsSummary {
  ScalarSummary mean_ms;
  ScalarSummary median_ms;
  ScalarSummary p95_ms;
  ScalarSummary cv;
};

struct ConcurrencyMetricsSummary {
  ScalarSummary duration_ms;
  ScalarSummary average;
  ScalarSummary peak;
  ScalarSummary p50;
  ScalarSummary p95;
  ScalarSummary cv;
};

struct EndToEndSpanSummary {
  bool available = false;
  std::string unavailable_reason;
  ConcurrencyMetricsSummary metrics;
};

struct DagConcurrencySummary {
  std::int64_t dag_index = 0;
  ScalarSummary task_work_ms;
  DurationStatisticsSummary task_duration;
  ConcurrencyMetricsSummary task_gpu_span;
};

struct CombinedConcurrencySummary {
  ScalarSummary task_work_ms;
  DurationStatisticsSummary task_duration;
  ConcurrencyMetricsSummary task_gpu_span;
  EndToEndSpanSummary end_to_end_span;
};

struct ConcurrencySummary {
  std::size_t sample_count = 0;
  std::vector<DagConcurrencySummary> dags;
  CombinedConcurrencySummary combined;
};

struct ConcurrencyAnalysis {
  bool available = false;
  std::string unavailable_reason;
  std::vector<ConcurrencySampleMetrics> samples;
  ConcurrencySummary summary;
};

struct DerivedMetrics {
  ParallelismAnalysis parallelism;
  ConcurrencyAnalysis concurrency;
};

DerivedMetrics derive_metrics(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples);

#endif
