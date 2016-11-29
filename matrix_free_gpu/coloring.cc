
#include <deal.II/base/graph_coloring.h>

#include "coloring.h"

using namespace dealii;

template <int dim, typename Iterator>
std::vector<types::global_dof_index>
GraphColoringWrapper<dim,Iterator>::get_conflict_indices (GraphColoringWrapper<dim,Iterator>::CellFilter const &cell,
                                                          const ConstraintMatrix &constraints)
{
  std::vector<types::global_dof_index> local_dof_indices(cell->get_fe().dofs_per_cell);
  cell->get_active_or_mg_dof_indices(local_dof_indices);

  constraints.resolve_indices(local_dof_indices);
  return local_dof_indices;
}

template <int dim, typename Iterator>
typename GraphColoringWrapper<dim,Iterator>::GraphType
GraphColoringWrapper<dim,Iterator>::make_graph_coloring(const Iterator &begin,
                                                        const Iterator &end,
                                                        const ConstraintMatrix    &constraints)
{
  CellFilter begin_filter(IteratorFilters::LocallyOwnedCell(),begin);
  CellFilter end_filter(IteratorFilters::LocallyOwnedCell(),end);

  typedef std_cxx11::function<std::vector<types::global_dof_index> (CellFilter const &)> function_type;
  const function_type &fun = static_cast<function_type> (std_cxx11::bind(&get_conflict_indices, std_cxx11::_1,constraints));

  return GraphColoring::make_graph_coloring(begin_filter,end_filter,fun);
}

// for active cells
template class GraphColoringWrapper<2,typename DoFHandler<2>::active_cell_iterator>;
template class GraphColoringWrapper<3,typename DoFHandler<3>::active_cell_iterator>;
// for multigrid levels
template class GraphColoringWrapper<2,typename DoFHandler<2>::level_cell_iterator>;
template class GraphColoringWrapper<3,typename DoFHandler<3>::level_cell_iterator>;
