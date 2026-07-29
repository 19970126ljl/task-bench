#!/usr/bin/env python3

import argparse
import json
import math
import os
import struct
import tempfile


class StableHash:
    def __init__(self):
        self.value = 14695981039346656037

    def add_u8(self, value):
        self.value ^= value & 0xFF
        self.value = (self.value * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def add_u64(self, value):
        value &= 0xFFFFFFFFFFFFFFFF
        for _ in range(8):
            self.add_u8(value)
            value >>= 8

    def add_i64(self, value):
        self.add_u64(value)

    def add_f64(self, value):
        value = float(value)
        if value == 0.0:
            value = 0.0
        self.add_u64(struct.unpack("<Q", struct.pack("<d", value))[0])

    def add_string(self, value):
        encoded = value.encode("utf-8")
        self.add_u64(len(encoded))
        for byte in encoded:
            self.add_u8(byte)

    def digest(self):
        return f"{self.value:016x}"


def add_bool(hasher, value):
    hasher.add_u8(1 if value else 0)


def compute_raw_data_hash(run):
    hasher = StableHash()
    hasher.add_string("cudastf-raw-data")
    hasher.add_string(compute_workload_config_hash(run))
    hasher.add_string(compute_execution_config_hash(run))
    hasher.add_string(compute_environment_hash(run))
    dags = run["dags"]
    hasher.add_u64(len(dags))
    for dag in dags:
        resources = dag["kernel"]["resources"]
        hasher.add_u64(len(resources))
        for resource in resources:
            hasher.add_i64(resource["device_id"])
            hasher.add_i64(resource["registers_per_thread"])
            hasher.add_u64(resource["static_shared_memory_bytes"])
            hasher.add_i64(resource["max_active_blocks_per_sm"])
    samples = run["samples"]
    hasher.add_u64(len(samples))
    for sample in samples:
        hasher.add_f64(sample["submission_ms"])
        hasher.add_f64(sample["dag_makespan_ms"])
        diagnostics = sample["diagnostics"]
        hasher.add_f64(diagnostics["setup_ms"])
        hasher.add_f64(diagnostics["sample_total_ms"])
        profile = sample["task_profile"]
        add_bool(hasher, profile is not None)
        if profile is None:
            continue
        hasher.add_u64(profile["cupti_timestamp_origin_ns"])
        contexts = profile["contexts"]
        hasher.add_u64(len(contexts))
        for context in contexts:
            hasher.add_u64(context["context_id"])
            hasher.add_string(context["label"])
            add_bool(hasher, context["has_gpu_activity"])
            hasher.add_u64(context["start_ns"])
            hasher.add_u64(context["end_ns"])
            hasher.add_f64(context["elapsed_ms"])
            hasher.add_u64(context["task_count"])
            hasher.add_u64(context["operation_count"])
            hasher.add_string(context["task_serialization"])
            regions = context["regions"]
            hasher.add_u64(len(regions))
            for region in regions:
                hasher.add_u64(region["region_id"])
                hasher.add_string(region["label"])
                add_bool(hasher, region["has_gpu_activity"])
                hasher.add_u64(region["start_ns"])
                hasher.add_u64(region["end_ns"])
                hasher.add_f64(region["elapsed_ms"])
                hasher.add_u64(region["task_count"])
                hasher.add_u64(region["operation_count"])
        tasks = profile["tasks"]
        hasher.add_u64(len(tasks))
        for task in tasks:
            hasher.add_i64(task["dag_index"])
            hasher.add_i64(task["timestep"])
            hasher.add_i64(task["point"])
            hasher.add_i64(task["configured_device"])
            hasher.add_u64(task["context_id"])
            hasher.add_u64(task["region_id"])
            hasher.add_i64(task["task_id"])
            hasher.add_string(task["symbol"])
            add_bool(hasher, task["has_gpu_activity"])
            hasher.add_u64(task["start_ns"])
            hasher.add_u64(task["end_ns"])
            hasher.add_f64(task["elapsed_ms"])
            hasher.add_u64(task["operation_count"])
            devices = task["device_timings"]
            hasher.add_u64(len(devices))
            for device in devices:
                hasher.add_i64(device["device_id"])
                hasher.add_u64(device["start_ns"])
                hasher.add_u64(device["end_ns"])
                hasher.add_f64(device["elapsed_ms"])
                hasher.add_u64(device["operation_count"])
    return hasher.digest()


def task_key(task):
    return (task["dag_index"], task["timestep"], task["point"])


def compute_topology_hash(dag):
    hasher = StableHash()
    hasher.add_string("task-bench-dag-topology")
    tasks = dag["task_table"]
    hasher.add_u64(len(tasks))
    for task in tasks:
        hasher.add_i64(task["timestep"])
        hasher.add_i64(task["point"])
        predecessors = task["predecessors"]
        hasher.add_u64(len(predecessors))
        for predecessor in predecessors:
            hasher.add_i64(predecessor["timestep"])
            hasher.add_i64(predecessor["point"])
    return hasher.digest()


def compute_workload_config_hash(run):
    config = run["run_config"]
    device_ids = config["device_ids"]
    if not device_ids:
        raise ValueError("execution configuration has no CUDA devices")
    hasher = StableHash()
    hasher.add_string("cudastf-workload-config")
    hasher.add_string(config["context"])
    hasher.add_string(config["logical_data_allocator"])
    hasher.add_i64(device_ids[0])
    if len(device_ids) > 1:
        hasher.add_string("multi-gpu-placement")
        hasher.add_u64(len(device_ids))
        for device_id in device_ids:
            hasher.add_i64(device_id)
        hasher.add_string(config["task_placement"])
    dags = run["dags"]
    hasher.add_u64(len(dags))
    for dag in dags:
        hasher.add_string(dag["topology_hash"])
        kernel = dag["kernel"]
        kernel_type = kernel["type"]
        if kernel_type not in {
            "empty", "busy_wait", "memory_bound", "compute_bound"
        }:
            raise ValueError(
                f"unsupported kernel type in execution identity: {kernel_type}"
            )
        hasher.add_string(kernel_type)
        if kernel_type == "busy_wait":
            hasher.add_i64(kernel["iterations"])
        elif kernel_type == "memory_bound":
            hasher.add_i64(kernel["iterations"])
            hasher.add_i64(kernel["samples"])
        elif kernel_type == "compute_bound":
            hasher.add_i64(kernel["iterations"])
            hasher.add_string(kernel["compute_data_type"])
        hasher.add_f64(kernel["imbalance"])
        hasher.add_u64(dag["output_bytes_per_task"])
        hasher.add_u64(dag["scratch_bytes_per_task"])
        hasher.add_i64(dag["nb_fields"])
        launch = kernel["launch"]
        hasher.add_i64(launch["blocks_per_task"])
        hasher.add_i64(launch["threads_per_block"])
        hasher.add_u64(launch["dynamic_shared_memory_bytes"])
    return hasher.digest()


def compute_execution_config_hash(run):
    config = run["run_config"]
    hasher = StableHash()
    hasher.add_string("cudastf-execution-config")
    hasher.add_string(compute_workload_config_hash(run))
    hasher.add_string(config["task_serialization"])
    hasher.add_string(config["task_profiler"])
    return hasher.digest()


def compute_environment_hash(run):
    hasher = StableHash()
    hasher.add_string("cudastf-environment")
    hasher.add_string(run["task_bench_revision"])
    add_bool(hasher, run["task_bench_worktree_dirty"])
    hasher.add_string(run["cccl_revision"])
    add_bool(hasher, run["cccl_worktree_dirty"])
    hasher.add_string(run["build"]["type"])
    hasher.add_string(run["build"]["cuda_arch_resolved"])
    hasher.add_i64(run["cuda_runtime_version"])
    hasher.add_i64(run["cuda_driver_version"])
    devices = run["devices"]
    hasher.add_u64(len(devices))
    for device in devices:
        hasher.add_string(device["name"])
        major, minor = device["compute_capability"].split(".", 1)
        hasher.add_i64(int(major))
        hasher.add_i64(int(minor))
        hasher.add_i64(device["sm_count"])
    return hasher.digest()


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def nearest_rank(values, fraction):
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[min(index, len(ordered) - 1)]


def population_cv(values, mean_value):
    if mean_value == 0.0:
        return 0.0
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance) / mean_value


def topology_fields(task_count, edge_count, widths):
    if not widths:
        raise ValueError("DAG topology has no ASAP levels")
    average = task_count / len(widths)
    return {
        "tasks": task_count,
        "dependency_edges": edge_count,
        "critical_path_length": len(widths),
        "parallelism": {
            "average": average,
            "peak": max(widths),
            "p50": median(widths),
            "p95": nearest_rank(widths, 0.95),
            "cv": population_cv(widths, average),
        },
    }


def analyze_topology(dags):
    dag_results = []
    combined_widths = []
    combined_tasks = 0
    combined_edges = 0
    for dag in dags:
        if compute_topology_hash(dag) != dag["topology_hash"]:
            raise ValueError(f"DAG {dag['dag_index']} topology hash mismatch")
        finishes = {}
        widths = []
        edge_count = 0
        for task in dag["task_table"]:
            if task["dag_index"] != dag["dag_index"]:
                raise ValueError("logical task DAG index mismatch")
            key = task_key(task)
            if key in finishes:
                raise ValueError(f"duplicate logical task {key}")
            start = 0
            for predecessor in task["predecessors"]:
                predecessor_key = (
                    task["dag_index"],
                    predecessor["timestep"],
                    predecessor["point"],
                )
                if predecessor_key not in finishes:
                    raise ValueError(f"missing or unordered predecessor {predecessor_key}")
                start = max(start, finishes[predecessor_key])
                edge_count += 1
            while len(widths) <= start:
                widths.append(0)
            widths[start] += 1
            finishes[key] = start + 1
        if edge_count != dag["dependency_edges"]:
            raise ValueError(f"DAG {dag['dag_index']} edge count mismatch")
        if len(dag["task_table"]) != dag["tasks"]:
            raise ValueError(f"DAG {dag['dag_index']} task count mismatch")
        fields = topology_fields(len(finishes), edge_count, widths)
        dag_results.append({"dag_index": dag["dag_index"], **fields})
        combined_tasks += len(finishes)
        combined_edges += edge_count
        while len(combined_widths) < len(widths):
            combined_widths.append(0)
        for index, width in enumerate(widths):
            combined_widths[index] += width
    return {
        "dags": dag_results,
        "combined": topology_fields(
            combined_tasks, combined_edges, combined_widths
        ),
    }


def analyze_kernel_capacity(run):
    devices = run["devices"]
    devices_by_id = {}
    for device in devices:
        device_id = device["device_id"]
        if device_id in devices_by_id:
            raise ValueError(f"duplicate CUDA device id {device_id}")
        if device["sm_count"] <= 0:
            raise ValueError(f"non-positive SM count for device {device_id}")
        devices_by_id[device_id] = device

    dag_results = []
    for dag in run["dags"]:
        blocks_per_task = dag["kernel"]["launch"]["blocks_per_task"]
        if blocks_per_task <= 0:
            raise ValueError("blocks per task is not positive")
        used_devices = {
            task["configured_device"] for task in dag["task_table"]
        }
        if not used_devices:
            raise ValueError(f"DAG {dag['dag_index']} has no assigned device")

        resources_by_device = {}
        for resource in dag["kernel"]["resources"]:
            device_id = resource["device_id"]
            if device_id in resources_by_device:
                raise ValueError(
                    f"duplicate kernel resource device id {device_id}"
                )
            resources_by_device[device_id] = resource

        device_results = []
        combined_resident_blocks = 0
        combined_saturation_tasks = 0
        for device in devices:
            device_id = device["device_id"]
            if device_id not in used_devices:
                continue
            if device_id not in resources_by_device:
                raise ValueError(
                    f"missing kernel resources for device {device_id}"
                )
            resource = resources_by_device[device_id]
            active_blocks = resource["max_active_blocks_per_sm"]
            if active_blocks <= 0:
                raise ValueError(
                    f"non-positive active blocks for device {device_id}"
                )
            resident_blocks = device["sm_count"] * active_blocks
            saturation_tasks = (
                resident_blocks + blocks_per_task - 1
            ) // blocks_per_task
            device_results.append(
                {
                    "device_id": device_id,
                    "sm_count": device["sm_count"],
                    "max_active_blocks_per_sm": active_blocks,
                    "resident_blocks": resident_blocks,
                    "occupancy_saturation_tasks": saturation_tasks,
                }
            )
            combined_resident_blocks += resident_blocks
            combined_saturation_tasks += saturation_tasks
        if len(device_results) != len(used_devices):
            raise ValueError(
                f"DAG {dag['dag_index']} uses an unknown CUDA device"
            )
        dag_results.append(
            {
                "dag_index": dag["dag_index"],
                "devices": device_results,
                "combined": {
                    "device_count": len(device_results),
                    "resident_blocks": combined_resident_blocks,
                    "occupancy_saturation_tasks": (
                        combined_saturation_tasks
                    ),
                },
            }
        )
    return {"dags": dag_results}


def duration_statistics(values):
    mean_value = sum(values) / len(values)
    return {
        "mean_ms": mean_value,
        "median_ms": median(values),
        "p95_ms": nearest_rank(values, 0.95),
        "cv": population_cv(values, mean_value),
    }


def model_asap_intervals(tasks, duration_by_task):
    finishes = {}
    intervals = []
    for task in tasks:
        key = task_key(task)
        if key not in duration_by_task:
            raise ValueError(f"missing duration for task {key}")
        duration = duration_by_task[key]
        if duration <= 0:
            raise ValueError(f"task duration is not positive for task {key}")
        start = 0
        for predecessor in task["predecessors"]:
            predecessor_key = (
                task["dag_index"],
                predecessor["timestep"],
                predecessor["point"],
            )
            if predecessor_key not in finishes:
                raise ValueError(
                    f"missing or unordered predecessor {predecessor_key}"
                )
            start = max(start, finishes[predecessor_key])
        finish = start + duration
        finishes[key] = finish
        intervals.append((start, finish))
    return intervals


def parallelism_statistics(intervals):
    events = {}
    work = 0.0
    critical_path = 0.0
    for start, finish in intervals:
        if not finish > start:
            raise ValueError("measured duration is not positive")
        work += finish - start
        critical_path = max(critical_path, finish)
        events[start] = events.get(start, 0) + 1
        events[finish] = events.get(finish, 0) - 1
    active = 0
    previous = min(events)
    duration_by_value = {}
    peak = 0
    for timestamp in sorted(events):
        duration = timestamp - previous
        if duration > 0.0:
            duration_by_value[active] = duration_by_value.get(active, 0.0) + duration
            peak = max(peak, active)
        active += events[timestamp]
        previous = timestamp
    if active != 0 or not critical_path > 0.0:
        raise ValueError("invalid parallelism interval events")
    average = work / critical_path

    def weighted_quantile(fraction):
        threshold = fraction * critical_path
        cumulative = 0.0
        for value in sorted(duration_by_value):
            cumulative += duration_by_value[value]
            if cumulative >= threshold:
                return float(value)
        return float(max(duration_by_value))

    variance = sum(
        duration * (value - average) ** 2
        for value, duration in duration_by_value.items()
    ) / critical_path
    return {
        "task_work_ms": work,
        "critical_path_ms": critical_path,
        "average": average,
        "peak": peak,
        "p50": weighted_quantile(0.50),
        "p95": weighted_quantile(0.95),
        "cv": math.sqrt(variance) / average if average else 0.0,
    }


def unavailable_parallelism(reason):
    return {"status": "unavailable", "reason": reason}


def unavailable_concurrency(reason):
    return {"status": "unavailable", "reason": reason}


def concurrency_histogram(intervals):
    if not intervals:
        raise ValueError("concurrency interval set is empty")
    events = {}
    task_work_ns = 0
    for start_ns, end_ns in intervals:
        if end_ns <= start_ns:
            raise ValueError("runtime task duration is not positive")
        task_work_ns += end_ns - start_ns
        events[start_ns] = events.get(start_ns, 0) + 1
        events[end_ns] = events.get(end_ns, 0) - 1

    timestamps = sorted(events)
    task_gpu_span_ns = timestamps[-1] - timestamps[0]
    if task_gpu_span_ns <= 0:
        raise ValueError("Task GPU span is not positive")

    duration_by_concurrency = {}
    peak = 0
    active = 0
    previous_ns = timestamps[0]
    for timestamp in timestamps:
        duration_ns = timestamp - previous_ns
        if duration_ns > 0:
            if active < 0:
                raise ValueError("invalid concurrency interval events")
            duration_by_concurrency[active] = (
                duration_by_concurrency.get(active, 0) + duration_ns
            )
            peak = max(peak, active)
        active += events[timestamp]
        if active < 0:
            raise ValueError("invalid concurrency interval events")
        previous_ns = timestamp
    if active != 0:
        raise ValueError("unbalanced concurrency interval events")
    return {
        "task_work_ns": task_work_ns,
        "task_gpu_span_ns": task_gpu_span_ns,
        "peak": peak,
        "duration_by_concurrency": duration_by_concurrency,
    }


def concurrency_span(histogram, duration_ms, duration_ns=None):
    if not math.isfinite(duration_ms) or duration_ms <= 0.0:
        raise ValueError("concurrency span is not positive")
    if duration_ns is None:
        duration_ns = duration_ms * 1.0e6
    task_gpu_span_ns = histogram["task_gpu_span_ns"]
    if duration_ns < task_gpu_span_ns:
        raise ValueError("concurrency span is shorter than Task GPU span")

    duration_by_concurrency = dict(histogram["duration_by_concurrency"])
    duration_by_concurrency[0] = (
        duration_by_concurrency.get(0, 0.0)
        + duration_ns - task_gpu_span_ns
    )
    average = histogram["task_work_ns"] / duration_ns

    def weighted_quantile(fraction):
        threshold = fraction * duration_ns
        cumulative = 0.0
        for value in sorted(duration_by_concurrency):
            cumulative += duration_by_concurrency[value]
            if cumulative >= threshold:
                return float(value)
        return float(max(duration_by_concurrency))

    variance = sum(
        duration * (value - average) ** 2
        for value, duration in duration_by_concurrency.items()
    ) / duration_ns
    return {
        "duration_ms": duration_ms,
        "average": average,
        "peak": histogram["peak"],
        "p50": weighted_quantile(0.50),
        "p95": weighted_quantile(0.95),
        "cv": math.sqrt(variance) / average if average else 0.0,
    }


def scalar_summary(values):
    return {"median": median(values), "p95": nearest_rank(values, 0.95)}


def duration_statistics_summary(values):
    return {
        name: scalar_summary([value[name] for value in values])
        for name in ("mean_ms", "median_ms", "p95_ms", "cv")
    }


def concurrency_span_summary(values):
    return {
        name: scalar_summary([value[name] for value in values])
        for name in ("duration_ms", "average", "peak", "p50", "p95", "cv")
    }


def analyze_parallelism(run):
    config = run["run_config"]
    if config["task_profiler"] != "enabled" or config["task_serialization"] != "enabled":
        return unavailable_parallelism(
            "parallelism requires task profiler and task serialization enabled"
        )
    samples = run["samples"]
    if not samples:
        return unavailable_parallelism("parallelism requires measured samples")

    dag_tasks = {}
    expected = {}
    for dag in run["dags"]:
        dag_tasks[dag["dag_index"]] = dag["task_table"]
        for task in dag["task_table"]:
            key = task_key(task)
            if key in expected:
                raise ValueError(f"duplicate logical task {key}")
            expected[key] = task
    observations = {key: [] for key in expected}

    for sample in samples:
        profile = sample["task_profile"]
        if profile is None:
            raise ValueError("profiled sample has no task profile")
        contexts = profile["contexts"]
        if len(contexts) != 1 or contexts[0]["task_serialization"] != "enabled":
            raise ValueError("parallelism sample is not marked as serialized")
        if len(profile["tasks"]) != len(expected):
            raise ValueError("profiled sample task count mismatch")
        if contexts[0]["task_count"] != len(expected):
            raise ValueError("profiled context task count mismatch")
        runtime_ids = set()
        seen = set()
        active_intervals = []
        for task in profile["tasks"]:
            key = task_key(task)
            runtime_id = (task["context_id"], task["task_id"])
            if task["context_id"] != contexts[0]["context_id"]:
                raise ValueError("task context id mismatch")
            if runtime_id in runtime_ids:
                raise ValueError("duplicate context/task id")
            runtime_ids.add(runtime_id)
            if key not in expected or key in seen:
                raise ValueError(f"unknown or duplicate logical task {key}")
            seen.add(key)
            if task["configured_device"] != expected[key]["configured_device"]:
                raise ValueError(f"configured device mismatch for task {key}")
            if not task["has_gpu_activity"]:
                return unavailable_parallelism(
                    "one or more tasks have no correlated GPU activity"
                )
            start_ns = task["start_ns"]
            end_ns = task["end_ns"]
            if end_ns <= start_ns or task["operation_count"] <= 0:
                return unavailable_parallelism(
                    "one or more tasks have a non-positive GPU duration"
                )
            expected_elapsed = (end_ns - start_ns) / 1.0e6
            if not math.isclose(
                task["elapsed_ms"], expected_elapsed,
                rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"elapsed time mismatch for task {key}")
            if not task["device_timings"]:
                raise ValueError(f"missing device timing for task {key}")
            if len(task["device_timings"]) != 1:
                raise ValueError(f"unexpected device timing count for task {key}")
            device_operation_count = 0
            for device in task["device_timings"]:
                if device["device_id"] != task["configured_device"]:
                    raise ValueError(f"observed device mismatch for task {key}")
                if (device["end_ns"] <= device["start_ns"] or
                        device["operation_count"] <= 0):
                    raise ValueError(f"invalid device timing for task {key}")
                device_elapsed = (device["end_ns"] - device["start_ns"]) / 1.0e6
                if not math.isclose(
                    device["elapsed_ms"], device_elapsed,
                    rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"device elapsed time mismatch for task {key}")
                device_operation_count += device["operation_count"]
            if device_operation_count != task["operation_count"]:
                raise ValueError(f"device operation count mismatch for task {key}")
            observations[key].append(expected_elapsed)
            active_intervals.append((start_ns, end_ns))
        if seen != set(expected):
            raise ValueError("profile is missing logical tasks")
        active_intervals.sort()
        for previous, current in zip(active_intervals, active_intervals[1:]):
            if previous[1] > current[0]:
                raise ValueError("serialized task GPU activity overlaps")

    measured_duration = {
        key: median(values) for key, values in observations.items()
    }
    dag_results = []
    combined_durations = []
    combined_intervals = []
    for dag in run["dags"]:
        task_results = []
        durations = []
        for task in dag["task_table"]:
            key = task_key(task)
            duration = measured_duration[key]
            durations.append(duration)
            combined_durations.append(duration)
            task_results.append(
                {
                    "dag_index": key[0],
                    "timestep": key[1],
                    "point": key[2],
                    "configured_device": task["configured_device"],
                    "measured_duration_ms": duration,
                }
            )
        intervals = model_asap_intervals(
            dag["task_table"], measured_duration
        )
        combined_intervals.extend(intervals)
        dag_results.append(
            {
                "dag_index": dag["dag_index"],
                "tasks": task_results,
                "task_duration": duration_statistics(durations),
                "parallelism": parallelism_statistics(intervals),
            }
        )
    return {
        "status": "available",
        "duration_aggregation": "median",
        "duration_definition": "gpu_activity_envelope",
        "dags": dag_results,
        "combined": {
            "task_duration": duration_statistics(combined_durations),
            "parallelism": parallelism_statistics(combined_intervals),
        },
    }


def analyze_concurrency(run):
    config = run["run_config"]
    if (config["task_profiler"] != "enabled" or
            config["task_serialization"] != "disabled"):
        return unavailable_concurrency(
            "concurrency requires task profiler enabled and task serialization disabled"
        )
    samples = run["samples"]
    if not samples:
        return unavailable_concurrency("concurrency requires measured samples")

    expected = {}
    dag_tasks = {}
    for dag in run["dags"]:
        dag_tasks[dag["dag_index"]] = dag["task_table"]
        for task in dag["task_table"]:
            key = task_key(task)
            if key in expected:
                raise ValueError(f"duplicate logical task {key}")
            expected[key] = task

    configured_devices = {}
    sample_results = []
    for sample_index, sample in enumerate(samples):
        profile = sample["task_profile"]
        if profile is None:
            raise ValueError("profiled sample has no task profile")
        contexts = profile["contexts"]
        if (len(contexts) != 1 or
                contexts[0]["task_serialization"] != "disabled"):
            raise ValueError(
                "concurrency sample is not marked as normal execution"
            )
        if len(profile["tasks"]) != len(expected):
            raise ValueError("profiled sample task count mismatch")
        if contexts[0]["task_count"] != len(expected):
            raise ValueError("profiled context task count mismatch")

        runtime_ids = set()
        tasks_by_key = {}
        for task in profile["tasks"]:
            key = task_key(task)
            runtime_id = (task["context_id"], task["task_id"])
            if task["context_id"] != contexts[0]["context_id"]:
                raise ValueError("task context id mismatch")
            if runtime_id in runtime_ids:
                raise ValueError("duplicate context/task id")
            runtime_ids.add(runtime_id)
            if key not in expected or key in tasks_by_key:
                raise ValueError(f"unknown or duplicate logical task {key}")
            if task["configured_device"] != expected[key]["configured_device"]:
                raise ValueError(f"configured device mismatch for task {key}")
            previous_device = configured_devices.setdefault(
                key, task["configured_device"]
            )
            if previous_device != task["configured_device"]:
                raise ValueError(f"configured device changed for task {key}")
            if not task["has_gpu_activity"]:
                return unavailable_concurrency(
                    "one or more tasks have no correlated GPU activity"
                )
            start_ns = task["start_ns"]
            end_ns = task["end_ns"]
            if end_ns <= start_ns or task["operation_count"] <= 0:
                return unavailable_concurrency(
                    "one or more tasks have a non-positive GPU duration"
                )
            expected_elapsed = (end_ns - start_ns) / 1.0e6
            if not math.isclose(
                task["elapsed_ms"], expected_elapsed,
                rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"elapsed time mismatch for task {key}")
            if len(task["device_timings"]) != 1:
                raise ValueError(f"unexpected device timing count for task {key}")
            device = task["device_timings"][0]
            if device["device_id"] != task["configured_device"]:
                raise ValueError(f"observed device mismatch for task {key}")
            if (device["end_ns"] <= device["start_ns"] or
                    device["operation_count"] <= 0):
                raise ValueError(f"invalid device timing for task {key}")
            device_elapsed = (
                device["end_ns"] - device["start_ns"]
            ) / 1.0e6
            if not math.isclose(
                device["elapsed_ms"], device_elapsed,
                rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"device elapsed time mismatch for task {key}")
            if device["operation_count"] != task["operation_count"]:
                raise ValueError(f"device operation count mismatch for task {key}")
            tasks_by_key[key] = task
        if set(tasks_by_key) != set(expected):
            raise ValueError("profile is missing logical tasks")

        dag_results = []
        combined_durations = []
        combined_intervals = []
        for dag in run["dags"]:
            durations = []
            intervals = []
            for logical_task in dag_tasks[dag["dag_index"]]:
                task = tasks_by_key[task_key(logical_task)]
                duration_ms = (task["end_ns"] - task["start_ns"]) / 1.0e6
                interval = (task["start_ns"], task["end_ns"])
                durations.append(duration_ms)
                intervals.append(interval)
                combined_durations.append(duration_ms)
                combined_intervals.append(interval)
            histogram = concurrency_histogram(intervals)
            task_gpu_span_ms = histogram["task_gpu_span_ns"] / 1.0e6
            dag_results.append(
                {
                    "dag_index": dag["dag_index"],
                    "task_work_ms": histogram["task_work_ns"] / 1.0e6,
                    "task_duration": duration_statistics(durations),
                    "task_gpu_span": concurrency_span(
                        histogram, task_gpu_span_ms,
                        histogram["task_gpu_span_ns"],
                    ),
                }
            )

        combined_histogram = concurrency_histogram(combined_intervals)
        combined_task_gpu_span_ms = (
            combined_histogram["task_gpu_span_ns"] / 1.0e6
        )
        end_to_end_ms = sample["dag_makespan_ms"]
        if not math.isfinite(end_to_end_ms) or end_to_end_ms <= 0.0:
            end_to_end_span = unavailable_concurrency(
                "End-to-end span is not positive"
            )
        elif end_to_end_ms * 1.0e6 < combined_histogram["task_gpu_span_ns"]:
            end_to_end_span = unavailable_concurrency(
                "End-to-end span is shorter than Task GPU span"
            )
        else:
            end_to_end_span = {
                "status": "available",
                **concurrency_span(combined_histogram, end_to_end_ms),
            }
        sample_results.append(
            {
                "sample_index": sample_index,
                "dags": dag_results,
                "combined": {
                    "task_work_ms": (
                        combined_histogram["task_work_ns"] / 1.0e6
                    ),
                    "task_duration": duration_statistics(combined_durations),
                    "task_gpu_span": concurrency_span(
                        combined_histogram, combined_task_gpu_span_ms,
                        combined_histogram["task_gpu_span_ns"],
                    ),
                    "end_to_end_span": end_to_end_span,
                },
            }
        )

    dag_summaries = []
    for dag_index, dag in enumerate(run["dags"]):
        values = [sample["dags"][dag_index] for sample in sample_results]
        if any(value["dag_index"] != dag["dag_index"] for value in values):
            raise ValueError("DAG order changed across concurrency samples")
        dag_summaries.append(
            {
                "dag_index": dag["dag_index"],
                "task_work_ms": scalar_summary(
                    [value["task_work_ms"] for value in values]
                ),
                "task_duration": duration_statistics_summary(
                    [value["task_duration"] for value in values]
                ),
                "task_gpu_span": concurrency_span_summary(
                    [value["task_gpu_span"] for value in values]
                ),
            }
        )

    combined_values = [sample["combined"] for sample in sample_results]
    end_to_end_values = [
        value["end_to_end_span"] for value in combined_values
    ]
    if all(value["status"] == "available" for value in end_to_end_values):
        end_to_end_summary = {
            "status": "available",
            **concurrency_span_summary(end_to_end_values),
        }
    else:
        end_to_end_summary = unavailable_concurrency(
            "one or more samples have an unavailable End-to-end span"
        )
    return {
        "status": "available",
        "duration_definition": "gpu_activity_envelope",
        "samples": sample_results,
        "summary": {
            "sample_count": len(sample_results),
            "dags": dag_summaries,
            "combined": {
                "task_work_ms": scalar_summary(
                    [value["task_work_ms"] for value in combined_values]
                ),
                "task_duration": duration_statistics_summary(
                    [value["task_duration"] for value in combined_values]
                ),
                "task_gpu_span": concurrency_span_summary(
                    [value["task_gpu_span"] for value in combined_values]
                ),
                "end_to_end_span": end_to_end_summary,
            },
        },
    }


def analyze(run):
    if run.get("format") != "cudastf-task-bench-run":
        raise ValueError("input is not a CUDASTF Task Bench run JSON")
    if run.get("schema_version") != 4:
        raise ValueError("unsupported run JSON schema version")
    for dag in run["dags"]:
        actual_topology_hash = compute_topology_hash(dag)
        if actual_topology_hash != dag.get("topology_hash"):
            raise ValueError(
                f"DAG {dag['dag_index']} topology hash mismatch"
            )
    actual_workload_hash = compute_workload_config_hash(run)
    if actual_workload_hash != run.get("workload_config_hash"):
        raise ValueError("workload config hash mismatch")
    actual_execution_hash = compute_execution_config_hash(run)
    if actual_execution_hash != run.get("execution_config_hash"):
        raise ValueError("execution config hash mismatch")
    actual_environment_hash = compute_environment_hash(run)
    if actual_environment_hash != run.get("environment_hash"):
        raise ValueError("environment hash mismatch")
    actual_hash = compute_raw_data_hash(run)
    if actual_hash != run.get("raw_data_hash"):
        raise ValueError(
            f"raw data hash mismatch: expected {run.get('raw_data_hash')}, got {actual_hash}"
        )
    topology = analyze_topology(run["dags"])
    return {
        "format": "cudastf-task-bench-analysis",
        "schema_version": 5,
        "backend": "cudastf",
        "source": {
            "task_bench_revision": run["task_bench_revision"],
            "task_bench_worktree_dirty": run["task_bench_worktree_dirty"],
            "workload_config_hash": run["workload_config_hash"],
            "execution_config_hash": run["execution_config_hash"],
            "environment_hash": run["environment_hash"],
            "raw_data_hash": run["raw_data_hash"],
            "dags": [
                {
                    "dag_index": dag["dag_index"],
                    "topology_hash": dag["topology_hash"],
                }
                for dag in run["dags"]
            ],
        },
        "topology": topology,
        "kernel_capacity": analyze_kernel_capacity(run),
        "derived_metrics": {
            "parallelism": analyze_parallelism(run),
            "concurrency": analyze_concurrency(run),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute CUDASTF Task Bench analysis from raw run JSON"
    )
    parser.add_argument("--input", required=True, help="raw run JSON")
    parser.add_argument("--output", required=True, help="analysis JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.input, "r", encoding="utf-8") as source:
        run = json.load(source)
    result = analyze(run)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".analysis-", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(result, destination, indent=2, sort_keys=False, allow_nan=False)
            destination.write("\n")
        os.replace(temporary, args.output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
