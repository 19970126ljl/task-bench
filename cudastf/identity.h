#ifndef TASK_BENCH_CUDASTF_IDENTITY_H
#define TASK_BENCH_CUDASTF_IDENTITY_H

#include <string>
#include <vector>

#include "arguments.h"
#include "expanded_dag.h"

std::string compute_topology_hash(const std::vector<DagTask> &tasks);

std::string compute_execution_config_hash(
    const RunConfig &run_config,
    const std::vector<GpuKernelConfig> &gpu_kernel_configs,
    const std::vector<ExpandedDag> &expanded_dags);

#endif
