#include "derived_metrics.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include "results.h"

namespace {

struct ModeledInterval {
  double start_ms;
  double finish_ms;
};

struct RuntimeInterval {
  std::uint64_t start_ns;
  std::uint64_t end_ns;
};

struct ConcurrencyHistogram {
  long double task_work_ns = 0.0L;
  long double task_gpu_span_ns = 0.0L;
  std::uint64_t peak = 0;
  std::map<std::uint64_t, long double> duration_by_concurrency;
};

double median(std::vector<double> values)
{
  if (values.empty()) throw std::logic_error("median requires values");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 != 0) return values[middle];
  return (values[middle - 1] + values[middle]) / 2.0;
}

double nearest_rank(std::vector<double> values, double fraction)
{
  if (values.empty()) throw std::logic_error("percentile requires values");
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
  return values[std::min(index, values.size() - 1)];
}

DurationStatistics summarize_durations(
    const std::vector<double> &durations)
{
  if (durations.empty()) {
    throw std::logic_error("measured duration set is empty");
  }
  DurationStatistics result;
  for (double duration : durations) result.mean_ms += duration;
  result.mean_ms /= static_cast<double>(durations.size());
  result.median_ms = median(durations);
  result.p95_ms = nearest_rank(durations, 0.95);
  if (result.mean_ms > 0.0) {
    double squared_error = 0.0;
    for (double duration : durations) {
      const double difference = duration - result.mean_ms;
      squared_error += difference * difference;
    }
    result.cv = std::sqrt(
        squared_error / static_cast<double>(durations.size())) /
        result.mean_ms;
  }
  return result;
}

ParallelismMetrics summarize_modeled_intervals(
    const std::vector<ModeledInterval> &intervals)
{
  if (intervals.empty()) {
    throw std::logic_error("parallelism interval set is empty");
  }

  ParallelismMetrics result;
  std::map<double, std::int64_t> events;
  for (const ModeledInterval &interval : intervals) {
    if (!(interval.finish_ms > interval.start_ms)) {
      throw std::logic_error("measured task duration is not positive");
    }
    result.task_work_ms += interval.finish_ms - interval.start_ms;
    result.critical_path_ms =
        std::max(result.critical_path_ms, interval.finish_ms);
    ++events[interval.start_ms];
    --events[interval.finish_ms];
  }
  if (!(result.critical_path_ms > 0.0)) {
    throw std::logic_error("weighted critical path is not positive");
  }

  std::map<std::uint64_t, double> duration_by_parallelism;
  std::int64_t active = 0;
  double previous_time = events.begin()->first;
  for (const auto &event : events) {
    const double duration = event.first - previous_time;
    if (duration > 0.0) {
      if (active < 0) {
        throw std::logic_error("invalid parallelism interval events");
      }
      const auto active_tasks = static_cast<std::uint64_t>(active);
      duration_by_parallelism[active_tasks] += duration;
      result.peak = std::max(result.peak, active_tasks);
    }
    active += event.second;
    previous_time = event.first;
  }
  if (active != 0) {
    throw std::logic_error("unbalanced parallelism interval events");
  }

  result.average = result.task_work_ms / result.critical_path_ms;
  const auto weighted_quantile = [&](double fraction) {
    const double threshold = fraction * result.critical_path_ms;
    double cumulative = 0.0;
    for (const auto &entry : duration_by_parallelism) {
      cumulative += entry.second;
      if (cumulative >= threshold) {
        return static_cast<double>(entry.first);
      }
    }
    return static_cast<double>(duration_by_parallelism.rbegin()->first);
  };
  result.p50 = weighted_quantile(0.50);
  result.p95 = weighted_quantile(0.95);

  if (result.average > 0.0) {
    double weighted_squared_error = 0.0;
    for (const auto &entry : duration_by_parallelism) {
      const double difference =
          static_cast<double>(entry.first) - result.average;
      weighted_squared_error +=
          entry.second * difference * difference;
    }
    result.cv = std::sqrt(
        weighted_squared_error / result.critical_path_ms) /
        result.average;
  }
  return result;
}

ConcurrencyHistogram make_concurrency_histogram(
    const std::vector<RuntimeInterval> &intervals)
{
  if (intervals.empty()) {
    throw std::logic_error("concurrency interval set is empty");
  }

  ConcurrencyHistogram result;
  std::map<std::uint64_t, std::int64_t> events;
  for (const RuntimeInterval &interval : intervals) {
    if (interval.end_ns <= interval.start_ns) {
      throw std::logic_error("runtime task duration is not positive");
    }
    result.task_work_ns +=
        static_cast<long double>(interval.end_ns - interval.start_ns);
    ++events[interval.start_ns];
    --events[interval.end_ns];
  }

  const std::uint64_t first_ns = events.begin()->first;
  const std::uint64_t last_ns = events.rbegin()->first;
  if (last_ns <= first_ns) {
    throw std::logic_error("Task GPU span is not positive");
  }
  result.task_gpu_span_ns =
      static_cast<long double>(last_ns - first_ns);

  std::int64_t active = 0;
  std::uint64_t previous_ns = first_ns;
  for (const auto &event : events) {
    if (event.first > previous_ns) {
      if (active < 0) {
        throw std::logic_error("invalid concurrency interval events");
      }
      const auto active_tasks = static_cast<std::uint64_t>(active);
      result.duration_by_concurrency[active_tasks] +=
          static_cast<long double>(event.first - previous_ns);
      result.peak = std::max(result.peak, active_tasks);
    }
    active += event.second;
    if (active < 0) {
      throw std::logic_error("invalid concurrency interval events");
    }
    previous_ns = event.first;
  }
  if (active != 0) {
    throw std::logic_error("unbalanced concurrency interval events");
  }
  return result;
}

ConcurrencyMetrics summarize_concurrency(
    const ConcurrencyHistogram &histogram,
    long double duration_ns, double duration_ms)
{
  if (!(duration_ms > 0.0) || !std::isfinite(duration_ms)) {
    throw std::logic_error("concurrency span is not positive");
  }
  if (duration_ns < histogram.task_gpu_span_ns) {
    throw std::logic_error(
        "concurrency span is shorter than Task GPU span");
  }

  std::map<std::uint64_t, long double> duration_by_concurrency =
      histogram.duration_by_concurrency;
  duration_by_concurrency[0] +=
      duration_ns - histogram.task_gpu_span_ns;

  ConcurrencyMetrics result;
  result.duration_ms = duration_ms;
  result.average = static_cast<double>(
      histogram.task_work_ns / duration_ns);
  result.peak = histogram.peak;
  const auto weighted_quantile = [&](long double fraction) {
    const long double threshold = fraction * duration_ns;
    long double cumulative = 0.0L;
    for (const auto &entry : duration_by_concurrency) {
      cumulative += entry.second;
      if (cumulative >= threshold) {
        return static_cast<double>(entry.first);
      }
    }
    return static_cast<double>(duration_by_concurrency.rbegin()->first);
  };
  result.p50 = weighted_quantile(0.50L);
  result.p95 = weighted_quantile(0.95L);

  if (result.average > 0.0) {
    long double weighted_squared_error = 0.0L;
    for (const auto &entry : duration_by_concurrency) {
      const long double difference =
          static_cast<long double>(entry.first) - result.average;
      weighted_squared_error +=
          entry.second * difference * difference;
    }
    result.cv = static_cast<double>(
        std::sqrt(weighted_squared_error / duration_ns) /
        result.average);
  }
  return result;
}

DagTaskKey key_for(const TaskCoordinates &coordinates)
{
  return {coordinates.dag_index, coordinates.timestep, coordinates.point};
}

ScalarSummary summarize_scalars(const std::vector<double> &values)
{
  return {median(values), nearest_rank(values, 0.95)};
}

template <typename T, typename Getter>
ScalarSummary summarize_field(
    const std::vector<const T *> &values, Getter getter)
{
  std::vector<double> observations;
  observations.reserve(values.size());
  for (const T *value : values) observations.push_back(getter(*value));
  return summarize_scalars(observations);
}

DurationStatisticsSummary summarize_duration_statistics(
    const std::vector<const DurationStatistics *> &values)
{
  DurationStatisticsSummary result;
  result.mean_ms = summarize_field(
      values, [](const DurationStatistics &value) { return value.mean_ms; });
  result.median_ms = summarize_field(
      values, [](const DurationStatistics &value) { return value.median_ms; });
  result.p95_ms = summarize_field(
      values, [](const DurationStatistics &value) { return value.p95_ms; });
  result.cv = summarize_field(
      values, [](const DurationStatistics &value) { return value.cv; });
  return result;
}

ConcurrencyMetricsSummary summarize_concurrency_metrics(
    const std::vector<const ConcurrencyMetrics *> &values)
{
  ConcurrencyMetricsSummary result;
  result.duration_ms = summarize_field(
      values, [](const ConcurrencyMetrics &value) {
        return value.duration_ms;
      });
  result.average = summarize_field(
      values, [](const ConcurrencyMetrics &value) {
        return value.average;
      });
  result.peak = summarize_field(
      values, [](const ConcurrencyMetrics &value) {
        return static_cast<double>(value.peak);
      });
  result.p50 = summarize_field(
      values, [](const ConcurrencyMetrics &value) { return value.p50; });
  result.p95 = summarize_field(
      values, [](const ConcurrencyMetrics &value) { return value.p95; });
  result.cv = summarize_field(
      values, [](const ConcurrencyMetrics &value) { return value.cv; });
  return result;
}

DagConcurrencySummary summarize_dag_concurrency(
    const std::vector<const DagConcurrencyMetrics *> &values)
{
  if (values.empty()) {
    throw std::logic_error("DAG concurrency summary requires samples");
  }
  DagConcurrencySummary result;
  result.dag_index = values.front()->dag_index;
  std::vector<const DurationStatistics *> durations;
  std::vector<const ConcurrencyMetrics *> concurrency_metrics;
  durations.reserve(values.size());
  concurrency_metrics.reserve(values.size());
  for (const DagConcurrencyMetrics *value : values) {
    if (value->dag_index != result.dag_index) {
      throw std::logic_error("DAG order changed across concurrency samples");
    }
    durations.push_back(&value->task_duration);
    concurrency_metrics.push_back(&value->task_gpu_span);
  }
  result.task_work_ms = summarize_field(
      values, [](const DagConcurrencyMetrics &value) {
        return value.task_work_ms;
      });
  result.task_duration = summarize_duration_statistics(durations);
  result.task_gpu_span =
      summarize_concurrency_metrics(concurrency_metrics);
  return result;
}

CombinedConcurrencySummary summarize_combined_concurrency(
    const std::vector<const CombinedConcurrencyMetrics *> &values)
{
  if (values.empty()) {
    throw std::logic_error("combined concurrency summary requires samples");
  }
  CombinedConcurrencySummary result;
  std::vector<const DurationStatistics *> durations;
  std::vector<const ConcurrencyMetrics *> task_gpu_metrics;
  std::vector<const ConcurrencyMetrics *> end_to_end_metrics;
  durations.reserve(values.size());
  task_gpu_metrics.reserve(values.size());
  end_to_end_metrics.reserve(values.size());
  bool all_end_to_end_available = true;
  for (const CombinedConcurrencyMetrics *value : values) {
    durations.push_back(&value->task_duration);
    task_gpu_metrics.push_back(&value->task_gpu_span);
    if (value->end_to_end_span.available) {
      end_to_end_metrics.push_back(&value->end_to_end_span.metrics);
    } else {
      all_end_to_end_available = false;
    }
  }
  result.task_work_ms = summarize_field(
      values, [](const CombinedConcurrencyMetrics &value) {
        return value.task_work_ms;
      });
  result.task_duration = summarize_duration_statistics(durations);
  result.task_gpu_span =
      summarize_concurrency_metrics(task_gpu_metrics);
  if (all_end_to_end_available) {
    result.end_to_end_span.available = true;
    result.end_to_end_span.metrics =
        summarize_concurrency_metrics(end_to_end_metrics);
  } else {
    result.end_to_end_span.unavailable_reason =
        "one or more samples have an unavailable End-to-end span";
  }
  return result;
}

ParallelismAnalysis derive_parallelism(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples)
{
  ParallelismAnalysis parallelism;
  if (run_config.task_profiler != CudaFeatureState::enabled ||
      run_config.task_serialization != CudaFeatureState::enabled) {
    parallelism.unavailable_reason =
        "parallelism requires task profiler and task serialization enabled";
    return parallelism;
  }
  if (samples.empty()) {
    parallelism.unavailable_reason =
        "parallelism requires measured samples";
    return parallelism;
  }

  std::map<DagTaskKey, std::vector<double>> observations;
  std::map<DagTaskKey, int> configured_devices;
  std::size_t expected_task_count = 0;
  for (const ExpandedDag &dag : expanded_dags) {
    expected_task_count += dag.tasks.size();
    for (const DagTask &task : dag.tasks) {
      observations.emplace(key_for(task.coordinates),
                           std::vector<double>{});
    }
  }

  for (const SampleResult &sample : samples) {
    if (!sample.task_profile.has_value()) {
      throw std::logic_error("profiled sample has no task profile");
    }
    if (sample.task_profile->contexts.size() != 1 ||
        sample.task_profile->contexts.front().task_serialization !=
            CudaFeatureState::enabled) {
      throw std::logic_error(
          "parallelism sample is not marked as serialized");
    }
    if (sample.task_profile->tasks.size() != expected_task_count) {
      throw std::logic_error("profiled sample task count mismatch");
    }
    std::map<DagTaskKey, bool> seen;
    for (const TaskProfileRecord &task : sample.task_profile->tasks) {
      auto observation = observations.find(task.key);
      if (observation == observations.end()) {
        throw std::logic_error("profile contains an unknown logical task");
      }
      if (!seen.emplace(task.key, true).second) {
        throw std::logic_error("profile contains a duplicate logical task");
      }
      auto device = configured_devices.emplace(
          task.key, task.configured_device);
      if (!device.second && device.first->second != task.configured_device) {
        throw std::logic_error("task placement changed across samples");
      }
      if (!task.has_gpu_activity) {
        parallelism.unavailable_reason =
            "one or more tasks have no correlated GPU activity";
        return parallelism;
      }
      if (task.end_ns <= task.start_ns) {
        parallelism.unavailable_reason =
            "one or more tasks have a non-positive GPU duration";
        return parallelism;
      }
      const double duration_ms =
          static_cast<double>(task.end_ns - task.start_ns) / 1.0e6;
      if (!(duration_ms > 0.0) || !std::isfinite(duration_ms)) {
        parallelism.unavailable_reason =
            "one or more tasks have a non-positive GPU duration";
        return parallelism;
      }
      observation->second.push_back(duration_ms);
    }
  }

  std::map<DagTaskKey, double> measured_duration;
  for (const auto &entry : observations) {
    if (entry.second.size() != samples.size()) {
      throw std::logic_error("measured task observation count mismatch");
    }
    measured_duration.emplace(entry.first, median(entry.second));
  }

  std::vector<double> combined_durations;
  std::vector<ModeledInterval> combined_intervals;
  parallelism.dags.reserve(expanded_dags.size());
  for (const ExpandedDag &dag : expanded_dags) {
    DagParallelismMetrics dag_result;
    dag_result.dag_index = dag.dag_index();
    std::map<DagTaskKey, double> finish_by_task;
    std::vector<double> durations;
    std::vector<ModeledInterval> intervals;
    for (const DagTask &task : dag.tasks) {
      const DagTaskKey key = key_for(task.coordinates);
      double start_ms = 0.0;
      for (const Predecessor &predecessor : task.predecessors) {
        const DagTaskKey predecessor_key{
            dag.dag_index(), predecessor.timestep, predecessor.point};
        const auto finish = finish_by_task.find(predecessor_key);
        if (finish == finish_by_task.end()) {
          throw std::logic_error(
              "parallelism predecessor is missing or not topologically ordered");
        }
        start_ms = std::max(start_ms, finish->second);
      }
      const double duration_ms = measured_duration.at(key);
      const double finish_ms = start_ms + duration_ms;
      finish_by_task.emplace(key, finish_ms);
      intervals.push_back({start_ms, finish_ms});
      durations.push_back(duration_ms);
      combined_intervals.push_back({start_ms, finish_ms});
      combined_durations.push_back(duration_ms);
      dag_result.tasks.push_back(
          {key, configured_devices.at(key), duration_ms});
    }
    dag_result.task_duration = summarize_durations(durations);
    dag_result.parallelism = summarize_modeled_intervals(intervals);
    parallelism.dags.push_back(std::move(dag_result));
  }

  parallelism.combined.task_duration =
      summarize_durations(combined_durations);
  parallelism.combined.parallelism =
      summarize_modeled_intervals(combined_intervals);
  parallelism.available = true;
  return parallelism;
}

ConcurrencyAnalysis derive_concurrency(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples)
{
  ConcurrencyAnalysis concurrency;
  if (run_config.task_profiler != CudaFeatureState::enabled ||
      run_config.task_serialization != CudaFeatureState::disabled) {
    concurrency.unavailable_reason =
        "concurrency requires task profiler enabled and task serialization disabled";
    return concurrency;
  }
  if (samples.empty()) {
    concurrency.unavailable_reason =
        "concurrency requires measured samples";
    return concurrency;
  }

  std::map<DagTaskKey, bool> expected_tasks;
  std::size_t expected_task_count = 0;
  for (const ExpandedDag &dag : expanded_dags) {
    expected_task_count += dag.tasks.size();
    for (const DagTask &task : dag.tasks) {
      if (!expected_tasks.emplace(key_for(task.coordinates), true).second) {
        throw std::logic_error("expanded DAGs contain a duplicate logical task");
      }
    }
  }

  std::map<DagTaskKey, int> configured_devices;
  concurrency.samples.reserve(samples.size());
  for (std::size_t sample_index = 0;
       sample_index < samples.size(); ++sample_index) {
    const SampleResult &sample = samples[sample_index];
    if (!sample.task_profile.has_value()) {
      throw std::logic_error("profiled sample has no task profile");
    }
    const TaskProfileSample &profile = *sample.task_profile;
    if (profile.contexts.size() != 1 ||
        profile.contexts.front().task_serialization !=
            CudaFeatureState::disabled) {
      throw std::logic_error(
          "concurrency sample is not marked as normal execution");
    }
    if (profile.tasks.size() != expected_task_count) {
      throw std::logic_error("profiled sample task count mismatch");
    }

    std::map<DagTaskKey, const TaskProfileRecord *> tasks_by_key;
    for (const TaskProfileRecord &task : profile.tasks) {
      if (expected_tasks.find(task.key) == expected_tasks.end()) {
        throw std::logic_error("profile contains an unknown logical task");
      }
      if (!tasks_by_key.emplace(task.key, &task).second) {
        throw std::logic_error("profile contains a duplicate logical task");
      }
      auto device = configured_devices.emplace(
          task.key, task.configured_device);
      if (!device.second && device.first->second != task.configured_device) {
        throw std::logic_error("task placement changed across samples");
      }
      if (!task.has_gpu_activity) {
        concurrency.unavailable_reason =
            "one or more tasks have no correlated GPU activity";
        concurrency.samples.clear();
        return concurrency;
      }
      if (task.end_ns <= task.start_ns) {
        concurrency.unavailable_reason =
            "one or more tasks have a non-positive GPU duration";
        concurrency.samples.clear();
        return concurrency;
      }
    }
    if (tasks_by_key.size() != expected_tasks.size()) {
      throw std::logic_error("profile is missing logical tasks");
    }

    ConcurrencySampleMetrics sample_metrics;
    sample_metrics.sample_index = sample_index;
    std::vector<double> combined_durations;
    std::vector<RuntimeInterval> combined_intervals;
    sample_metrics.dags.reserve(expanded_dags.size());
    for (const ExpandedDag &dag : expanded_dags) {
      DagConcurrencyMetrics dag_metrics;
      dag_metrics.dag_index = dag.dag_index();
      std::vector<double> durations;
      std::vector<RuntimeInterval> intervals;
      durations.reserve(dag.tasks.size());
      intervals.reserve(dag.tasks.size());
      for (const DagTask &dag_task : dag.tasks) {
        const TaskProfileRecord &task =
            *tasks_by_key.at(key_for(dag_task.coordinates));
        const std::uint64_t duration_ns = task.end_ns - task.start_ns;
        const double duration_ms =
            static_cast<double>(duration_ns) / 1.0e6;
        durations.push_back(duration_ms);
        intervals.push_back({task.start_ns, task.end_ns});
        combined_durations.push_back(duration_ms);
        combined_intervals.push_back({task.start_ns, task.end_ns});
      }
      const ConcurrencyHistogram histogram =
          make_concurrency_histogram(intervals);
      const double task_gpu_span_ms = static_cast<double>(
          histogram.task_gpu_span_ns / 1.0e6L);
      dag_metrics.task_work_ms = static_cast<double>(
          histogram.task_work_ns / 1.0e6L);
      dag_metrics.task_duration = summarize_durations(durations);
      dag_metrics.task_gpu_span =
          summarize_concurrency(
              histogram, histogram.task_gpu_span_ns,
              task_gpu_span_ms);
      sample_metrics.dags.push_back(std::move(dag_metrics));
    }

    const ConcurrencyHistogram combined_histogram =
        make_concurrency_histogram(combined_intervals);
    const double combined_task_gpu_span_ms = static_cast<double>(
        combined_histogram.task_gpu_span_ns / 1.0e6L);
    sample_metrics.combined.task_work_ms = static_cast<double>(
        combined_histogram.task_work_ns / 1.0e6L);
    sample_metrics.combined.task_duration =
        summarize_durations(combined_durations);
    sample_metrics.combined.task_gpu_span = summarize_concurrency(
        combined_histogram, combined_histogram.task_gpu_span_ns,
        combined_task_gpu_span_ms);

    const double end_to_end_ms = sample.performance.dag_makespan_ms;
    if (!(end_to_end_ms > 0.0) || !std::isfinite(end_to_end_ms)) {
      sample_metrics.combined.end_to_end_span.unavailable_reason =
          "End-to-end span is not positive";
    } else if (static_cast<long double>(end_to_end_ms) * 1.0e6L <
               combined_histogram.task_gpu_span_ns) {
      sample_metrics.combined.end_to_end_span.unavailable_reason =
          "End-to-end span is shorter than Task GPU span";
    } else {
      sample_metrics.combined.end_to_end_span.available = true;
      sample_metrics.combined.end_to_end_span.metrics =
          summarize_concurrency(
              combined_histogram,
              static_cast<long double>(end_to_end_ms) * 1.0e6L,
              end_to_end_ms);
    }
    concurrency.samples.push_back(std::move(sample_metrics));
  }

  concurrency.summary.sample_count = concurrency.samples.size();
  concurrency.summary.dags.reserve(expanded_dags.size());
  for (std::size_t dag_index = 0;
       dag_index < expanded_dags.size(); ++dag_index) {
    std::vector<const DagConcurrencyMetrics *> values;
    values.reserve(concurrency.samples.size());
    for (const ConcurrencySampleMetrics &sample : concurrency.samples) {
      values.push_back(&sample.dags.at(dag_index));
    }
    concurrency.summary.dags.push_back(
        summarize_dag_concurrency(values));
  }
  std::vector<const CombinedConcurrencyMetrics *> combined_values;
  combined_values.reserve(concurrency.samples.size());
  for (const ConcurrencySampleMetrics &sample : concurrency.samples) {
    combined_values.push_back(&sample.combined);
  }
  concurrency.summary.combined =
      summarize_combined_concurrency(combined_values);
  concurrency.available = true;
  return concurrency;
}

}  // namespace

DerivedMetrics derive_metrics(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples)
{
  DerivedMetrics result;
  result.parallelism =
      derive_parallelism(run_config, expanded_dags, samples);
  result.concurrency =
      derive_concurrency(run_config, expanded_dags, samples);
  return result;
}
