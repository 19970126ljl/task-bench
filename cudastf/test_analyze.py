#!/usr/bin/env python3

import copy
import unittest

import analyze


def make_task(dag_index, timestep, point, configured_device, predecessors):
    return {
        "dag_index": dag_index,
        "timestep": timestep,
        "point": point,
        "configured_device": configured_device,
        "iterations": 1,
        "predecessors": [
            {"timestep": predecessor[0], "point": predecessor[1]}
            for predecessor in predecessors
        ],
    }


def make_profile_task(task, task_id, start_ns, duration_ms):
    duration_ns = int(duration_ms * 1.0e6)
    end_ns = start_ns + duration_ns
    return {
        "dag_index": task["dag_index"],
        "timestep": task["timestep"],
        "point": task["point"],
        "configured_device": task["configured_device"],
        "context_id": 7,
        "region_id": 0,
        "task_id": task_id,
        "symbol": f"task-{task_id}",
        "has_gpu_activity": True,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "elapsed_ms": duration_ms,
        "operation_count": 1,
        "device_timings": [
            {
                "device_id": task["configured_device"],
                "start_ns": start_ns,
                "end_ns": end_ns,
                "elapsed_ms": duration_ms,
                "operation_count": 1,
            }
        ],
    }


def make_sample(tasks, durations, reverse):
    profile_tasks = []
    start_ns = 1000
    for index, (task, duration) in enumerate(zip(tasks, durations), 1):
        record = make_profile_task(task, index, start_ns, duration)
        profile_tasks.append(record)
        start_ns = record["end_ns"] + 1000
    if reverse:
        profile_tasks.reverse()
    context_end = max(task["end_ns"] for task in profile_tasks)
    return {
        "submission_ms": 0.5,
        "dag_makespan_ms": 20.0,
        "diagnostics": {"setup_ms": 0.25, "sample_total_ms": 21.0},
        "task_profile": {
            "cupti_timestamp_origin_ns": 100,
            "contexts": [
                {
                    "context_id": 7,
                    "label": "",
                    "has_gpu_activity": True,
                    "start_ns": 1000,
                    "end_ns": context_end,
                    "elapsed_ms": (context_end - 1000) / 1.0e6,
                    "task_count": len(tasks),
                    "operation_count": len(tasks),
                    "task_serialization": "enabled",
                    "regions": [],
                }
            ],
            "tasks": profile_tasks,
        },
    }


def make_run():
    tasks = [
        make_task(0, 0, 0, 0, []),
        make_task(0, 0, 1, 1, []),
        make_task(0, 1, 0, 0, [(0, 0)]),
    ]
    dag = {
        "dag_index": 0,
        "timesteps": 2,
        "max_width": 2,
        "dependence": "stencil_1d",
        "radix": 3,
        "period": -1,
        "fraction_connected": 1.0,
        "nb_fields": 2,
        "output_bytes_per_task": 16,
        "scratch_bytes_per_task": 0,
        "kernel": {
            "type": "compute_bound",
            "iterations": 128,
            "samples": 0,
            "imbalance": 0.0,
            "compute_data_type": "fp32",
            "launch": {
                "blocks_per_task": 2,
                "threads_per_block": 64,
                "dynamic_shared_memory_bytes": 0,
            },
            "resources": [
                {
                    "device_id": 0,
                    "registers_per_thread": 32,
                    "static_shared_memory_bytes": 0,
                    "max_active_blocks_per_sm": 4,
                },
                {
                    "device_id": 1,
                    "registers_per_thread": 32,
                    "static_shared_memory_bytes": 0,
                    "max_active_blocks_per_sm": 5,
                },
            ],
        },
        "tasks": len(tasks),
        "dependency_edges": 1,
        "task_table": tasks,
    }
    dag["topology_hash"] = analyze.compute_topology_hash(dag)
    run = {
        "format": "cudastf-task-bench-run",
        "schema_version": 5,
        "backend": "cudastf",
        "task_bench_revision": "fixture",
        "task_bench_worktree_dirty": False,
        "cccl_revision": "fixture-cccl",
        "cccl_worktree_dirty": False,
        "build": {
            "type": "release",
            "cuda_arch_option": "native",
            "cuda_arch_resolved": "sm_100",
        },
        "cuda_runtime_version": 13000,
        "cuda_driver_version": 13000,
        "devices": [
            {
                "device_id": 0,
                "name": "fixture-gpu",
                "uuid": "GPU-fixture-0",
                "compute_capability": "10.0",
                "sm_count": 10,
            },
            {
                "device_id": 1,
                "name": "fixture-gpu",
                "uuid": "GPU-fixture-1",
                "compute_capability": "10.0",
                "sm_count": 12,
            },
        ],
        "run_config": {
            "device_ids": [0, 1],
            "task_placement": "block",
            "context": "stream",
            "logical_data_allocator": "cached",
            "stream_pool_size_per_device": 2,
            "warmup_samples": 0,
            "measured_samples": 2,
            "task_profiler": "enabled",
            "task_serialization": "enabled",
        },
        "dags": [dag],
        "samples": [
            make_sample(tasks, [1.0, 3.0, 2.0], False),
            make_sample(tasks, [3.0, 5.0, 4.0], True),
        ],
    }
    run["workload_config_hash"] = analyze.compute_workload_config_hash(run)
    run["execution_config_hash"] = analyze.compute_execution_config_hash(run)
    run["environment_hash"] = analyze.compute_environment_hash(run)
    run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
    return run


def set_sample_intervals(sample, intervals, dag_makespan_ms, origin_ns):
    by_task_id = {task["task_id"]: task for task in sample["task_profile"]["tasks"]}
    for task_id, (start_ns, end_ns) in enumerate(intervals, 1):
        task = by_task_id[task_id]
        elapsed_ms = (end_ns - start_ns) / 1.0e6
        task["start_ns"] = start_ns
        task["end_ns"] = end_ns
        task["elapsed_ms"] = elapsed_ms
        device = task["device_timings"][0]
        device["start_ns"] = start_ns
        device["end_ns"] = end_ns
        device["elapsed_ms"] = elapsed_ms
    profile = sample["task_profile"]
    profile["cupti_timestamp_origin_ns"] = origin_ns
    context = profile["contexts"][0]
    context["task_serialization"] = "disabled"
    context["start_ns"] = min(start for start, _ in intervals)
    context["end_ns"] = max(end for _, end in intervals)
    context["elapsed_ms"] = (
        context["end_ns"] - context["start_ns"]
    ) / 1.0e6
    sample["dag_makespan_ms"] = dag_makespan_ms


def make_normal_run():
    run = make_run()
    run["run_config"]["task_serialization"] = "disabled"
    set_sample_intervals(
        run["samples"][0],
        [(1_000_000, 3_000_000),
         (6_000_000, 8_000_000),
         (8_000_000, 10_000_000)],
        12.0,
        100,
    )
    set_sample_intervals(
        run["samples"][1],
        [(1_000_000_001, 1_004_000_001),
         (1_001_000_001, 1_005_000_001),
         (1_004_000_001, 1_008_000_001)],
        10.0,
        1_000_000_000_000,
    )
    run["execution_config_hash"] = analyze.compute_execution_config_hash(run)
    run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
    return run


class AnalyzeTest(unittest.TestCase):
    def test_parallelism_and_topology(self):
        result = analyze.analyze(make_run())
        self.assertEqual(result["schema_version"], 5)
        self.assertEqual(result["topology"]["dags"][0]["critical_path_length"], 2)
        capacity = result["kernel_capacity"]["dags"][0]
        self.assertEqual(
            [device["resident_blocks"] for device in capacity["devices"]],
            [40, 60],
        )
        self.assertEqual(
            [
                device["occupancy_saturation_tasks"]
                for device in capacity["devices"]
            ],
            [20, 30],
        )
        self.assertEqual(capacity["combined"]["device_count"], 2)
        self.assertEqual(capacity["combined"]["resident_blocks"], 100)
        self.assertEqual(
            capacity["combined"]["occupancy_saturation_tasks"], 50
        )
        parallelism_analysis = result["derived_metrics"]["parallelism"]
        self.assertEqual(parallelism_analysis["status"], "available")
        self.assertEqual(
            [
                task["measured_duration_ms"]
                for task in parallelism_analysis["dags"][0]["tasks"]
            ],
            [2.0, 4.0, 3.0],
        )
        parallelism = parallelism_analysis["dags"][0]["parallelism"]
        self.assertAlmostEqual(parallelism["task_work_ms"], 9.0)
        self.assertAlmostEqual(parallelism["critical_path_ms"], 5.0)
        self.assertAlmostEqual(parallelism["average"], 1.8)
        self.assertEqual(parallelism["peak"], 2)
        self.assertEqual(parallelism["p50"], 2.0)
        self.assertEqual(parallelism["p95"], 2.0)
        self.assertAlmostEqual(parallelism["cv"], 2.0 / 9.0)

    def test_hash_rejects_changed_raw_timing(self):
        run = make_run()
        changed = copy.deepcopy(run)
        changed["samples"][0]["task_profile"]["tasks"][0]["end_ns"] += 1
        with self.assertRaisesRegex(ValueError, "raw data hash mismatch"):
            analyze.analyze(changed)

    def test_schema_4_run_remains_supported(self):
        run = make_run()
        run["schema_version"] = 4
        del run["run_config"]["stream_pool_size_per_device"]
        run["workload_config_hash"] = analyze.compute_workload_config_hash(run)
        run["execution_config_hash"] = analyze.compute_execution_config_hash(run)
        run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
        result = analyze.analyze(run)
        self.assertEqual(result["schema_version"], 5)

    def test_normal_profile_has_no_parallelism_analysis(self):
        run = make_normal_run()
        result = analyze.analyze(run)
        self.assertEqual(
            result["derived_metrics"]["parallelism"]["status"],
            "unavailable",
        )

    def test_normal_concurrency(self):
        run = make_normal_run()
        run["run_config"]["warmup_samples"] = 3
        result = analyze.analyze(run)
        concurrency = result["derived_metrics"]["concurrency"]
        self.assertEqual(concurrency["status"], "available")
        self.assertEqual(len(concurrency["samples"]), 2)
        self.assertEqual(concurrency["summary"]["sample_count"], 2)

        first = concurrency["samples"][0]["combined"]
        self.assertAlmostEqual(first["task_work_ms"], 6.0)
        self.assertAlmostEqual(first["task_gpu_span"]["duration_ms"], 9.0)
        self.assertAlmostEqual(first["task_gpu_span"]["average"], 2.0 / 3.0)
        self.assertEqual(first["task_gpu_span"]["peak"], 1)
        self.assertEqual(first["task_gpu_span"]["p50"], 1.0)
        self.assertEqual(first["end_to_end_span"]["status"], "available")
        self.assertAlmostEqual(first["end_to_end_span"]["average"], 0.5)
        self.assertEqual(first["end_to_end_span"]["p50"], 0.0)

        summary = concurrency["summary"]["combined"]
        self.assertAlmostEqual(summary["task_work_ms"]["median"], 9.0)
        self.assertAlmostEqual(summary["task_work_ms"]["p95"], 12.0)
        self.assertAlmostEqual(summary["task_gpu_span"]["peak"]["median"], 1.5)
        self.assertEqual(summary["end_to_end_span"]["status"], "available")
        self.assertAlmostEqual(
            summary["end_to_end_span"]["duration_ms"]["median"], 11.0
        )

    def test_short_end_to_end_span_preserves_gpu_metrics(self):
        run = make_normal_run()
        run["samples"][0]["dag_makespan_ms"] = 8.0
        run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
        concurrency = analyze.analyze(run)["derived_metrics"]["concurrency"]
        self.assertEqual(concurrency["status"], "available")
        self.assertEqual(
            concurrency["samples"][0]["combined"]["end_to_end_span"]["status"],
            "unavailable",
        )

    def test_normal_task_without_gpu_activity_is_unavailable(self):
        run = make_normal_run()
        task = run["samples"][0]["task_profile"]["tasks"][0]
        task["has_gpu_activity"] = False
        task["start_ns"] = 0
        task["end_ns"] = 0
        task["elapsed_ms"] = 0.0
        task["operation_count"] = 0
        task["device_timings"] = []
        run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
        concurrency = analyze.analyze(run)["derived_metrics"]["concurrency"]
        self.assertEqual(concurrency["status"], "unavailable")
        self.assertIn("no correlated GPU activity", concurrency["reason"])

    def test_duplicate_logical_task_is_rejected(self):
        run = make_run()
        tasks = run["samples"][0]["task_profile"]["tasks"]
        tasks[1]["dag_index"] = tasks[0]["dag_index"]
        tasks[1]["timestep"] = tasks[0]["timestep"]
        tasks[1]["point"] = tasks[0]["point"]
        run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
        with self.assertRaisesRegex(ValueError, "unknown or duplicate"):
            analyze.analyze(run)

    def test_configuration_and_environment_hashes_are_recomputed(self):
        mutations = [
            ("serialization", lambda run: run["run_config"].__setitem__(
                "task_serialization", "disabled"), "execution config"),
            ("profiler", lambda run: run["run_config"].__setitem__(
                "task_profiler", "disabled"), "execution config"),
            ("kernel", lambda run: run["dags"][0]["kernel"].__setitem__(
                "iterations", 256), "workload config"),
            ("placement", lambda run: run["run_config"].__setitem__(
                "task_placement", "cyclic"), "workload config"),
            ("stream pool", lambda run: run["run_config"].__setitem__(
                "stream_pool_size_per_device", 3), "workload config"),
            ("build", lambda run: run["build"].__setitem__(
                "type", "debug"), "environment"),
            ("gpu model", lambda run: run["devices"][0].__setitem__(
                "name", "different-gpu"), "environment"),
            ("sm count", lambda run: run["devices"][0].__setitem__(
                "sm_count", 20), "environment"),
        ]
        for name, mutate, expected in mutations:
            with self.subTest(name=name):
                changed = make_run()
                mutate(changed)
                with self.assertRaisesRegex(ValueError, expected):
                    analyze.analyze(changed)

    def test_raw_hash_rejects_changed_kernel_resources(self):
        changed = make_run()
        changed["dags"][0]["kernel"]["resources"][0][
            "max_active_blocks_per_sm"
        ] += 1
        with self.assertRaisesRegex(ValueError, "raw data hash mismatch"):
            analyze.analyze(changed)

    def test_dag_order_is_bound_to_workload_hash(self):
        run = make_run()
        second = copy.deepcopy(run["dags"][0])
        second["dag_index"] = 1
        second["task_table"] = [make_task(1, 0, 0, 0, [])]
        second["tasks"] = 1
        second["dependency_edges"] = 0
        second["topology_hash"] = analyze.compute_topology_hash(second)
        run["dags"].append(second)
        run["workload_config_hash"] = analyze.compute_workload_config_hash(run)
        run["execution_config_hash"] = analyze.compute_execution_config_hash(run)
        run["raw_data_hash"] = analyze.compute_raw_data_hash(run)
        run["dags"].reverse()
        with self.assertRaisesRegex(ValueError, "workload config"):
            analyze.analyze(run)

    def test_gpu_uuid_is_provenance_not_compatibility(self):
        run = make_run()
        original = run["environment_hash"]
        run["devices"][0]["uuid"] = "GPU-replacement"
        self.assertEqual(analyze.compute_environment_hash(run), original)


if __name__ == "__main__":
    unittest.main()
