#include "data_access.h"

DataAccess output_data_access(const ExpandedDag &dag, const DagTask &task)
{
  return {{dag.dag_index(),
           task.coordinates.timestep % dag.task_graph.nb_fields,
           task.coordinates.point},
          DataAccessMode::write};
}

DataAccess input_data_access(const ExpandedDag &dag,
                             const Predecessor &predecessor)
{
  return {{dag.dag_index(),
           predecessor.timestep % dag.task_graph.nb_fields,
           predecessor.point},
          DataAccessMode::read};
}
