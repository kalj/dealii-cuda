
#include <deal.II/base/graph_coloring.h>

#include "coloring.h"

using namespace dealii;

template <int dim>
std::vector<types::global_dof_index>
GraphColoringWrapper<dim>::get_conflict_indices (GraphColoringWrapper<dim>::CellFilter const &cell,
                                                 const ConstraintMatrix &constraints)
{
  std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().dofs_per_cell);
  cell->get_active_or_mg_dof_indices(local_dof_indices);

  constraints.resolve_indices(local_dof_indices);
  return local_dof_indices;
}

template <int dim>
typename GraphColoringWrapper<dim>::GraphType
GraphColoringWrapper<dim>::make_graph_coloring(const DoFHandler<dim>     &dof_handler,
                                               const ConstraintMatrix    &constraints)
{
  CellFilter begin(IteratorFilters::LocallyOwnedCell(),dof_handler.begin_active());
  CellFilter end(IteratorFilters::LocallyOwnedCell(),dof_handler.end());

  typedef std_cxx11::function<std::vector<types::global_dof_index> (CellFilter const &)> function_type;
  const function_type &fun = static_cast<function_type> (std_cxx11::bind(&get_conflict_indices, std_cxx11::_1,constraints));

  return GraphColoring::make_graph_coloring(begin,end,fun);
}

template class GraphColoringWrapper<2>;
template class GraphColoringWrapper<3>;
