#ifndef BMOP_COMMON_H
#define BMOP_COMMON_H


#include <deal.II/grid/tria.h>
#include "poisson_common.h"

template <int dim>
void pseudo_adaptive_refinement(dealii::Triangulation<dim> &triangulation,
                                domain_case_t domain,
                                int n_ref)
{
  triangulation.refine_global (std::max(n_ref-2,0));
  {
    float reduction = 0;
    if(domain == BALL)
      if(dim==2)
        reduction = 0.021;
      else
        reduction = 0.04;
    else if(dim==3)
      reduction = 0.015;
    // the radii in this refinement are adjusted such that we hit similar number
    // of dofs as the globally refined case on a similar situation

    // refine elements inside disk of radius 0.55
    for (typename dealii::Triangulation<dim>::active_cell_iterator
           cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned() &&
          cell->center().norm() < 0.55-reduction)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();

    // refine elements in annulus with radius 0.3 < r < 0.42
    for (typename dealii::Triangulation<dim>::active_cell_iterator
           cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned() &&
          cell->center().norm() > 0.3+reduction &&
          cell->center().norm() < 0.42-reduction)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();

    // refine elements in annulus with radius 0.335 < r < 0.41 (or 0.388)
    for (typename dealii::Triangulation<dim>::active_cell_iterator
           cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned() &&
          cell->center().norm() > 0.32+reduction &&
          cell->center().norm() <  0.41-reduction)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();


    {
      Point<dim> offset;
      for (unsigned int d=0; d<dim; ++d)
        offset[d] = -0.1*(d+1);
      for (typename
             dealii::Triangulation<dim>::active_cell_iterator
             cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().distance(offset) > 0.17+reduction &&
            cell->center().distance(offset) < 0.33-reduction)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();

      for (typename
             dealii::Triangulation<dim>::active_cell_iterator
             cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().distance(offset) > 0.21+reduction &&
            cell->center().distance(offset) < 0.31-reduction)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }

  }
}


template <int dim>
void bmop_setup_mesh(dealii::Triangulation<dim> &triangulation,
                     domain_case_t domain,
                     bool pseudo_adaptive_grid,
                     int n_ref)
{
  create_domain(triangulation,domain);

  if(pseudo_adaptive_grid)
    pseudo_adaptive_refinement(triangulation,domain,n_ref);
  else
    triangulation.refine_global (n_ref);
}


#endif
