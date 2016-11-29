/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)coloring.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _COLORING_H
#define _COLORING_H

#include <deal.II/base/types.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/lac/constraint_matrix.h>


template <int dim, typename Iterator>
struct GraphColoringWrapper {
public:
  typedef dealii::FilteredIterator<Iterator> CellFilter;
  typedef std::vector<std::vector<CellFilter>> GraphType;

  static GraphType make_graph_coloring(const Iterator     &begin,
                                       const Iterator     &end,
                                       const dealii::ConstraintMatrix    &constraints);
private:
  static std::vector<dealii::types::global_dof_index> get_conflict_indices (CellFilter const               &cell,
                                                                            const dealii::ConstraintMatrix &constraints);
};


#endif /* _COLORING_H */
