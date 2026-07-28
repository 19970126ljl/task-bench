#include "derived_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

#include "results.h"

namespace {

struct Interval {
  double start_ms;
  double finish_ms;
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

ParallelismMetrics summarize_intervals(
    const std::vector<Interval> &intervals)
{
  if (intervals.empty()) {
    throw std::logic_error("parallelism interval set is empty");
  }

  ParallelismMetrics result;
  std::map<double, std::int64_t> events;
  for (const Interval &interval : intervals) {
    if (!(interval.finish_ms > interval.start_ms)) {
      throw std::logic_error("measured task duration is not positive");
    }
    result.work_ms += interval.finish_ms - interval.start_ms;
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

  result.average = result.work_ms / result.critical_path_ms;
  const auto weighted_quantile =
      [&](double fraction) {
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

DagTaskKey key_for(const TaskCoordinates &coordinates)
{
  return {coordinates.dag_index, coordinates.timestep, coordinates.point};
}

}  // namespace

DerivedMetrics derive_metrics(
    const RunConfig &run_config,
    const std::vector<ExpandedDag> &expanded_dags,
    const std::vector<SampleResult> &samples)
{
  DerivedMetrics result;
  ParallelismAnalysis &parallelism = result.parallelism;
  if (run_config.task_profiler != CudaFeatureState::enabled ||
      run_config.task_serialization != CudaFeatureState::enabled) {
    parallelism.unavailable_reason =
        "parallelism requires task profiler and task serialization enabled";
    return result;
  }
  if (samples.empty()) {
    parallelism.unavailable_reason =
        "parallelism requires measured samples";
    return result;
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
        return result;
      }
      if (task.end_ns <= task.start_ns) {
        parallelism.unavailable_reason =
            "one or more tasks have a non-positive GPU duration";
        return result;
      }
      const double duration_ms =
          static_cast<double>(task.end_ns - task.start_ns) / 1.0e6;
      if (!(duration_ms > 0.0) || !std::isfinite(duration_ms)) {
        parallelism.unavailable_reason =
            "one or more tasks have a non-positive GPU duration";
        return result;
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
  std::vector<Interval> combined_intervals;
  parallelism.dags.reserve(expanded_dags.size());
  for (const ExpandedDag &dag : expanded_dags) {
    DagParallelismMetrics dag_result;
    dag_result.dag_index = dag.dag_index();
    std::map<DagTaskKey, double> finish_by_task;
    std::vector<double> durations;
    std::vector<Interval> intervals;
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
    dag_result.parallelism = summarize_intervals(intervals);
    parallelism.dags.push_back(std::move(dag_result));
  }

  parallelism.combined_task_duration =
      summarize_durations(combined_durations);
  parallelism.combined_parallelism =
      summarize_intervals(combined_intervals);
  parallelism.available = true;
  return result;
}
