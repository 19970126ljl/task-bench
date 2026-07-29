#!/usr/bin/env python3

import copy
import unittest

import analyze
import compare as comparison
from test_analyze import make_normal_run, make_profile_task, make_run, make_task


def refresh_hashes(run):
    for dag in run["dags"]:
        dag["topology_hash"] = analyze.compute_topology_hash(dag)
    run["workload_config_hash"] = analyze.compute_workload_config_hash(run)
    run["execution_config_hash"] = analyze.compute_execution_config_hash(run)
    run["environment_hash"] = analyze.compute_environment_hash(run)
    run["raw_data_hash"] = analyze.compute_raw_data_hash(run)


def add_independent_dag(run, durations_ms, serialized):
    logical_task = make_task(1, 0, 0, 0, [])
    dag = copy.deepcopy(run["dags"][0])
    dag.update(
        {
            "dag_index": 1,
            "timesteps": 1,
            "max_width": 1,
            "dependence": "trivial",
            "tasks": 1,
            "dependency_edges": 0,
            "task_table": [logical_task],
        }
    )
    run["dags"].append(dag)

    for sample_index, sample in enumerate(run["samples"]):
        profile = sample["task_profile"]
        context = profile["contexts"][0]
        existing = profile["tasks"]
        if serialized:
            start_ns = max(task["end_ns"] for task in existing) + 1_000
        else:
            start_ns = min(task["start_ns"] for task in existing) + 1_000_000
        task = make_profile_task(
            logical_task,
            max(existing_task["task_id"] for existing_task in existing) + 1,
            start_ns,
            durations_ms[sample_index],
        )
        existing.insert(0, task)
        context["task_count"] += 1
        context["operation_count"] += 1
        context["start_ns"] = min(
            context["start_ns"], task["start_ns"]
        )
        context["end_ns"] = max(context["end_ns"], task["end_ns"])
        context["elapsed_ms"] = (
            context["end_ns"] - context["start_ns"]
        ) / 1.0e6
    refresh_hashes(run)


class CompareTest(unittest.TestCase):
    def test_core_cross_mode_metrics(self):
        result = comparison.compare(make_run(), make_normal_run())

        self.assertEqual(result["format"], "cudastf-task-bench-comparison")
        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["summary"]["sample_count"], 2)
        self.assertEqual(result["source"]["serialized"]["measured_samples"], 2)
        self.assertEqual(result["source"]["normal"]["measured_samples"], 2)

        first = result["samples"][0]
        first_dag = first["dags"][0]
        self.assertAlmostEqual(first_dag["task_work"]["serialized_ms"], 9.0)
        self.assertAlmostEqual(first_dag["task_work"]["normal_ms"], 6.0)
        self.assertAlmostEqual(first_dag["task_work"]["efficiency"], 1.5)
        self.assertAlmostEqual(first_dag["ideal_time_ms"], 5.0)
        self.assertAlmostEqual(first_dag["average_parallelism"], 1.8)
        task_gpu = first_dag["task_gpu_span"]
        self.assertAlmostEqual(task_gpu["actual_time_ms"], 9.0)
        self.assertAlmostEqual(task_gpu["average_concurrency"], 2.0 / 3.0)
        self.assertAlmostEqual(
            task_gpu["concurrency_to_parallelism"], 10.0 / 27.0
        )
        self.assertAlmostEqual(
            task_gpu["ideal_to_actual_time"], 5.0 / 9.0
        )
        end_to_end = first["combined"]["end_to_end_span"]
        self.assertEqual(end_to_end["status"], "available")
        self.assertAlmostEqual(end_to_end["actual_time_ms"], 12.0)
        self.assertAlmostEqual(end_to_end["average_concurrency"], 0.5)
        self.assertAlmostEqual(
            end_to_end["concurrency_to_parallelism"], 5.0 / 18.0
        )
        self.assertAlmostEqual(end_to_end["ideal_to_actual_time"], 5.0 / 12.0)

        second = result["samples"][1]
        self.assertAlmostEqual(
            second["dags"][0]["task_work"]["efficiency"], 0.75
        )
        self.assertAlmostEqual(
            second["combined"]["task_gpu_span"][
                "concurrency_to_parallelism"
            ],
            5.0 / 6.0,
        )
        self.assertAlmostEqual(
            second["combined"]["end_to_end_span"][
                "concurrency_to_parallelism"
            ],
            2.0 / 3.0,
        )

        tasks = result["tasks"]
        self.assertEqual(
            [(task["timestep"], task["point"]) for task in tasks],
            [(0, 0), (0, 1), (1, 0)],
        )
        self.assertEqual(
            [task["task_duration"]["serialized_ms"] for task in tasks],
            [2.0, 4.0, 3.0],
        )
        self.assertAlmostEqual(
            tasks[0]["task_duration"]["normal_ms"]["median"], 3.0
        )
        self.assertAlmostEqual(
            tasks[0]["task_duration"]["efficiency"]["median"], 0.75
        )
        self.assertAlmostEqual(
            tasks[0]["task_duration"]["efficiency"]["p95"], 1.0
        )

        summary = result["summary"]["combined"]
        self.assertAlmostEqual(
            summary["task_work"]["efficiency"]["median"], 1.125
        )
        self.assertAlmostEqual(
            summary["task_work"]["efficiency"]["p95"], 1.5
        )
        self.assertEqual(summary["end_to_end_span"]["status"], "available")
        self.assertAlmostEqual(
            summary["end_to_end_span"]["ideal_to_actual_time"]["p95"],
            0.5,
        )

        for sample in result["samples"]:
            for metrics in sample["dags"] + [sample["combined"]]:
                span = metrics["task_gpu_span"]
                self.assertAlmostEqual(
                    span["ideal_to_actual_time"],
                    metrics["task_work"]["efficiency"]
                    * span["concurrency_to_parallelism"],
                )
            end_to_end = sample["combined"]["end_to_end_span"]
            self.assertAlmostEqual(
                end_to_end["ideal_to_actual_time"],
                sample["combined"]["task_work"]["efficiency"]
                * end_to_end["concurrency_to_parallelism"],
            )

    def test_profile_task_order_does_not_define_identity(self):
        normal = make_normal_run()
        for sample in normal["samples"]:
            sample["task_profile"]["tasks"].reverse()
        refresh_hashes(normal)
        result = comparison.compare(make_run(), normal)
        self.assertEqual(len(result["tasks"]), 3)
        self.assertEqual(len(result["samples"]), 2)

    def test_different_measured_sample_counts_are_allowed(self):
        serialized = make_run()
        serialized["samples"] = serialized["samples"][:1]
        serialized["run_config"]["measured_samples"] = 1
        refresh_hashes(serialized)
        result = comparison.compare(serialized, make_normal_run())
        self.assertEqual(result["source"]["serialized"]["measured_samples"], 1)
        self.assertEqual(result["source"]["normal"]["measured_samples"], 2)
        self.assertEqual(result["summary"]["sample_count"], 2)

    def test_multi_dag_combined_metrics(self):
        serialized = make_run()
        normal = make_normal_run()
        add_independent_dag(serialized, [5.0, 7.0], serialized=True)
        add_independent_dag(normal, [3.0, 6.0], serialized=False)

        result = comparison.compare(serialized, normal)
        self.assertEqual(len(result["summary"]["dags"]), 2)
        self.assertEqual(len(result["samples"][0]["dags"]), 2)
        self.assertAlmostEqual(
            result["samples"][0]["combined"]["task_work"]["efficiency"],
            15.0 / 9.0,
        )
        self.assertAlmostEqual(
            result["samples"][0]["combined"]["average_parallelism"],
            2.5,
        )
        self.assertAlmostEqual(
            result["samples"][0]["combined"]["task_gpu_span"][
                "concurrency_to_parallelism"
            ],
            0.4,
        )

    def test_unavailable_end_to_end_preserves_gpu_metrics(self):
        normal = make_normal_run()
        normal["samples"][0]["dag_makespan_ms"] = 8.0
        refresh_hashes(normal)
        result = comparison.compare(make_run(), normal)

        first = result["samples"][0]["combined"]
        self.assertGreater(
            first["task_gpu_span"]["concurrency_to_parallelism"], 0.0
        )
        self.assertEqual(first["end_to_end_span"]["status"], "unavailable")
        self.assertEqual(
            result["summary"]["combined"]["end_to_end_span"]["status"],
            "unavailable",
        )

    def test_pairing_rejects_incompatible_inputs(self):
        cases = []

        changed_workload = make_normal_run()
        changed_workload["dags"][0]["kernel"]["iterations"] += 1
        refresh_hashes(changed_workload)
        cases.append(("workload", changed_workload, "workload configuration"))

        changed_environment = make_normal_run()
        changed_environment["devices"][0]["name"] = "different-gpu"
        refresh_hashes(changed_environment)
        cases.append(("environment", changed_environment, "environment"))

        changed_device = make_normal_run()
        logical_task = changed_device["dags"][0]["task_table"][1]
        logical_task["configured_device"] = 0
        for sample in changed_device["samples"]:
            task = next(
                task for task in sample["task_profile"]["tasks"]
                if analyze.task_key(task) == (0, 0, 1)
            )
            task["configured_device"] = 0
            task["device_timings"][0]["device_id"] = 0
        refresh_hashes(changed_device)
        cases.append(("device", changed_device, "task keys, or devices"))

        changed_capacity = make_normal_run()
        changed_capacity["dags"][0]["kernel"]["resources"][0][
            "max_active_blocks_per_sm"
        ] += 1
        refresh_hashes(changed_capacity)
        cases.append(("capacity", changed_capacity, "kernel capacity"))

        for name, normal, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    comparison.compare(make_run(), normal)

    def test_pairing_requires_expected_execution_modes(self):
        with self.assertRaisesRegex(
                ValueError, "serialized input requires task serialization"):
            normal = make_normal_run()
            comparison.compare(normal, normal)


if __name__ == "__main__":
    unittest.main()
