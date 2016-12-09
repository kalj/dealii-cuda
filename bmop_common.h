#ifndef BMOP_COMMON_H
#define BMOP_COMMON_H


#include <deal.II/grid/tria.h>
#include "poisson_common.h"


template <int dim>
void mark_cells_in_annulus(dealii::Triangulation<dim> &triangulation,
                           double R,
                           double r=0.0,
                           const dealii::Point<dim> center=dealii::Point<dim>())
{
  for (typename
         dealii::Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active(); cell != triangulation.end(); ++cell) {
    if (cell->is_locally_owned() &&
        cell->center().distance(center) > r &&
        cell->center().distance(center) < R) {
      cell->set_refine_flag();
    }
  }
}


template <int dim>
void mark_cells_on_shell(dealii::Triangulation<dim> &triangulation,
                         double R,
                         const dealii::Point<dim> center=dealii::Point<dim>())
{
  const int nverts = 1<<dim;

  for (typename
         dealii::Triangulation<dim>::active_cell_iterator
         cell=triangulation.begin_active(); cell != triangulation.end(); ++cell) {
    if (cell->is_locally_owned()) {
      int ninside = 0;

      for(int i = 0; i < nverts; ++i)
        ninside += (cell->vertex(i).distance(center) < R);

      if(ninside!=0 && ninside !=nverts)
        cell->set_refine_flag();
    }
  }
}

template <int dim>
void pseudo_adaptive_refinement(dealii::Triangulation<dim> &triangulation,
                                domain_case_t domain,
                                int n_ref)
{

  n_ref = std::max(n_ref-2,0);

  triangulation.refine_global (n_ref);

  float reduction = 0;
  if(dim==2) {
    if(domain == BALL)
      reduction = 0.025;
    else
      reduction = 0.005;
  }
  else {
    if(domain == BALL)
      reduction = 0.04;
    else
      reduction = 0.015;
  }

  // the radii in this refinement are adjusted such that we hit similar number
  // of dofs as the globally refined case on a similar situation

  // refine elements inside disk of radius 0.55
  mark_cells_in_annulus(triangulation,0.55-reduction);
  triangulation.execute_coarsening_and_refinement();

  // refine elements in annulus with radius 0.3 < r < 0.42
  mark_cells_in_annulus(triangulation,0.42-reduction,0.3+reduction);
  triangulation.execute_coarsening_and_refinement();

  // refine elements in annulus with radius 0.335 < r < 0.41 (or 0.388)
  mark_cells_in_annulus(triangulation,0.41-reduction,0.32+reduction);
  triangulation.execute_coarsening_and_refinement();

  dealii::Point<dim> offset;
  for (unsigned int d=0; d<dim; ++d)
    offset[d] = -0.1*(d+1);

  mark_cells_in_annulus(triangulation,0.33-reduction,0.17+reduction,offset);
  triangulation.execute_coarsening_and_refinement();

  mark_cells_in_annulus(triangulation,0.31-reduction,0.21+reduction,offset);
  triangulation.execute_coarsening_and_refinement();

  if(dim==2) {
    int nshells=4;
    for(int s = 0; s < nshells; ++s) {
      mark_cells_on_shell(triangulation,0.25,offset);
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
