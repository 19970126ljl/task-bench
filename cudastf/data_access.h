#ifndef TASK_BENCH_CUDASTF_DATA_ACCESS_H
#define TASK_BENCH_CUDASTF_DATA_ACCESS_H

#include <cstdint>

#include "expanded_dag.h"

enum class DataAccessMode {
  write,
  read,
};

struct DataId {
  std::int64_t dag_index;
  std::int64_t field;
  std::int64_t point;
};

struct DataAccess {
  DataId data;
  DataAccessMode mode;
};

DataAccess output_data_access(const ExpandedDag &dag, const DagTask &task);
DataAccess input_data_access(const ExpandedDag &dag,
                             const Predecessor &predecessor);

#endif
