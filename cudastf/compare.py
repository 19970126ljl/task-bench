#!/usr/bin/env python3

import argparse
import json
import os
import tempfile

import analyze


def unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def task_identity(run):
    return [
        (
            dag["dag_index"],
            dag["topology_hash"],
            [
                (analyze.task_key(task), task["configured_device"])
                for task in dag["task_table"]
            ],
        )
        for dag in run["dags"]
    ]


def validate_pair(serialized_run, normal_run, serialized_analysis,
                  normal_analysis):
    if serialized_run["workload_config_hash"] != normal_run[
            "workload_config_hash"]:
        raise ValueError("run pair has different workload configuration hashes")
    if serialized_run["environment_hash"] != normal_run["environment_hash"]:
        raise ValueError("run pair has different environment hashes")
    if serialized_run["run_config"]["task_profiler"] != "enabled":
        raise ValueError("serialized input requires task profiler enabled")
    if serialized_run["run_config"]["task_serialization"] != "enabled":
        raise ValueError("serialized input requires task serialization enabled")
    if normal_run["run_config"]["task_profiler"] != "enabled":
        raise ValueError("normal input requires task profiler enabled")
    if normal_run["run_config"]["task_serialization"] != "disabled":
        raise ValueError("normal input requires task serialization disabled")
    for label, run in (
            ("serialized", serialized_run), ("normal", normal_run)):
        if run["run_config"]["measured_samples"] != len(run["samples"]):
            raise ValueError(f"{label} input measured sample count mismatch")
    if task_identity(serialized_run) != task_identity(normal_run):
        raise ValueError(
            "run pair has different DAG order, topology, task keys, or devices"
        )
    if serialized_analysis["kernel_capacity"] != normal_analysis[
            "kernel_capacity"]:
        raise ValueError("run pair has different kernel capacity")

    parallelism = serialized_analysis["derived_metrics"]["parallelism"]
    if parallelism["status"] != "available":
        raise ValueError(
            "serialized input parallelism is unavailable: "
            + parallelism["reason"]
        )
    concurrency = normal_analysis["derived_metrics"]["concurrency"]
    if concurrency["status"] != "available":
        raise ValueError(
            "normal input concurrency is unavailable: "
            + concurrency["reason"]
        )


def serialized_durations(serialized_analysis):
    result = {}
    parallelism = serialized_analysis["derived_metrics"]["parallelism"]
    for dag in parallelism["dags"]:
        for task in dag["tasks"]:
            key = analyze.task_key(task)
            if key in result:
                raise ValueError(f"duplicate serialized task {key}")
            result[key] = task["measured_duration_ms"]
    return result


def normal_sample_tasks(sample):
    result = {}
    for task in sample["task_profile"]["tasks"]:
        key = analyze.task_key(task)
        if key in result:
            raise ValueError(f"duplicate normal task {key}")
        result[key] = task
    return result


def span_metrics(span, average_parallelism, ideal_time_ms):
    return {
        "actual_time_ms": span["duration_ms"],
        "average_concurrency": span["average"],
        "concurrency_to_parallelism": (
            span["average"] / average_parallelism
        ),
        "ideal_to_actual_time": ideal_time_ms / span["duration_ms"],
    }


def sample_metrics(serialized_metrics, normal_metrics, include_end_to_end):
    serialized_work_ms = serialized_metrics["task_work_ms"]
    normal_work_ms = normal_metrics["task_work_ms"]
    ideal_time_ms = serialized_metrics["critical_path_ms"]
    average_parallelism = serialized_metrics["average"]
    result = {
        "task_work": {
            "serialized_ms": serialized_work_ms,
            "normal_ms": normal_work_ms,
            "efficiency": serialized_work_ms / normal_work_ms,
        },
        "ideal_time_ms": ideal_time_ms,
        "average_parallelism": average_parallelism,
        "task_gpu_span": span_metrics(
            normal_metrics["task_gpu_span"],
            average_parallelism,
            ideal_time_ms,
        ),
    }
    if include_end_to_end:
        end_to_end = normal_metrics["end_to_end_span"]
        if end_to_end["status"] == "available":
            result["end_to_end_span"] = {
                "status": "available",
                **span_metrics(
                    end_to_end, average_parallelism, ideal_time_ms
                ),
            }
        else:
            result["end_to_end_span"] = unavailable(end_to_end["reason"])
    return result


def span_summary(values):
    return {
        name: analyze.scalar_summary([value[name] for value in values])
        for name in (
            "actual_time_ms",
            "average_concurrency",
            "concurrency_to_parallelism",
            "ideal_to_actual_time",
        )
    }


def metrics_summary(values, include_end_to_end):
    first = values[0]
    result = {
        "task_work": {
            "serialized_ms": first["task_work"]["serialized_ms"],
            "normal_ms": analyze.scalar_summary(
                [value["task_work"]["normal_ms"] for value in values]
            ),
            "efficiency": analyze.scalar_summary(
                [value["task_work"]["efficiency"] for value in values]
            ),
        },
        "ideal_time_ms": first["ideal_time_ms"],
        "average_parallelism": first["average_parallelism"],
        "task_gpu_span": span_summary(
            [value["task_gpu_span"] for value in values]
        ),
    }
    if include_end_to_end:
        end_to_end_values = [value["end_to_end_span"] for value in values]
        if all(value["status"] == "available" for value in end_to_end_values):
            result["end_to_end_span"] = {
                "status": "available",
                **span_summary(end_to_end_values),
            }
        else:
            result["end_to_end_span"] = unavailable(
                "one or more samples have an unavailable End-to-end span"
            )
    return result


def source_fields(run):
    return {
        "execution_config_hash": run["execution_config_hash"],
        "raw_data_hash": run["raw_data_hash"],
        "measured_samples": run["run_config"]["measured_samples"],
    }


def compare(serialized_run, normal_run):
    serialized_analysis = analyze.analyze(serialized_run)
    normal_analysis = analyze.analyze(normal_run)
    validate_pair(
        serialized_run, normal_run, serialized_analysis, normal_analysis
    )

    serialized_by_key = serialized_durations(serialized_analysis)
    normal_observations = {key: [] for key in serialized_by_key}
    efficiency_observations = {key: [] for key in serialized_by_key}
    parallelism = serialized_analysis["derived_metrics"]["parallelism"]
    serialized_dags = {
        dag["dag_index"]: dag for dag in parallelism["dags"]
    }
    normal_concurrency = normal_analysis["derived_metrics"]["concurrency"]

    sample_results = []
    for sample_index, (sample, concurrency_sample) in enumerate(zip(
            normal_run["samples"], normal_concurrency["samples"])):
        if concurrency_sample["sample_index"] != sample_index:
            raise ValueError("normal concurrency sample order mismatch")
        tasks_by_key = normal_sample_tasks(sample)
        if set(tasks_by_key) != set(serialized_by_key):
            raise ValueError("run pair has different profiled task keys")

        for key, task in tasks_by_key.items():
            duration_ns = task["end_ns"] - task["start_ns"]
            duration_ms = duration_ns / 1.0e6
            normal_observations[key].append(duration_ms)
            efficiency_observations[key].append(
                serialized_by_key[key] / duration_ms
            )

        dag_results = []
        for dag_index, dag in enumerate(normal_run["dags"]):
            serialized_dag = serialized_dags[dag["dag_index"]]
            concurrency_dag = concurrency_sample["dags"][dag_index]
            if concurrency_dag["dag_index"] != dag["dag_index"]:
                raise ValueError("normal concurrency DAG order mismatch")
            dag_results.append({
                "dag_index": dag["dag_index"],
                **sample_metrics(
                    serialized_dag["parallelism"],
                    concurrency_dag,
                    include_end_to_end=False,
                ),
            })
        combined = sample_metrics(
            parallelism["combined"]["parallelism"],
            concurrency_sample["combined"],
            include_end_to_end=True,
        )
        sample_results.append(
            {
                "sample_index": sample_index,
                "dags": dag_results,
                "combined": combined,
            }
        )

    task_results = []
    expected_tasks = {
        analyze.task_key(task): task
        for dag in serialized_run["dags"] for task in dag["task_table"]
    }
    for key in serialized_by_key:
        task = expected_tasks[key]
        task_results.append(
            {
                "dag_index": key[0],
                "timestep": key[1],
                "point": key[2],
                "configured_device": task["configured_device"],
                "task_duration": {
                    "serialized_ms": serialized_by_key[key],
                    "normal_ms": analyze.scalar_summary(
                        normal_observations[key]
                    ),
                    "efficiency": analyze.scalar_summary(
                        efficiency_observations[key]
                    ),
                },
            }
        )

    dag_summaries = []
    for dag_index, dag in enumerate(normal_run["dags"]):
        values = [sample["dags"][dag_index] for sample in sample_results]
        if any(value["dag_index"] != dag["dag_index"] for value in values):
            raise ValueError("comparison DAG order changed across samples")
        dag_summaries.append(
            {
                "dag_index": dag["dag_index"],
                **metrics_summary(values, include_end_to_end=False),
            }
        )

    combined_values = [sample["combined"] for sample in sample_results]
    combined_summary = metrics_summary(
        combined_values, include_end_to_end=True
    )

    return {
        "format": "cudastf-task-bench-comparison",
        "schema_version": 1,
        "backend": "cudastf",
        "source": {
            "workload_config_hash": serialized_run["workload_config_hash"],
            "environment_hash": serialized_run["environment_hash"],
            "serialized": source_fields(serialized_run),
            "normal": source_fields(normal_run),
            "dags": [
                {
                    "dag_index": dag["dag_index"],
                    "topology_hash": dag["topology_hash"],
                }
                for dag in serialized_run["dags"]
            ],
        },
        "duration_definition": "gpu_activity_envelope",
        "serialized_duration_aggregation": "median",
        "tasks": task_results,
        "samples": sample_results,
        "summary": {
            "sample_count": len(sample_results),
            "dags": dag_summaries,
            "combined": combined_summary,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare profiled serialized and normal CUDASTF Task Bench runs"
        )
    )
    parser.add_argument(
        "--serialized", required=True, help="profiled serialized raw run JSON"
    )
    parser.add_argument(
        "--normal", required=True, help="profiled normal raw run JSON"
    )
    parser.add_argument("--output", required=True, help="comparison JSON")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.serialized, "r", encoding="utf-8") as source:
        serialized_run = json.load(source)
    with open(args.normal, "r", encoding="utf-8") as source:
        normal_run = json.load(source)
    result = compare(serialized_run, normal_run)

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".comparison-", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(
                result, destination, indent=2, sort_keys=False, allow_nan=False
            )
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
