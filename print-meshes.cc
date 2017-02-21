/* ---------------------------------------------------------------------
 * $Id$
 *
 * Copyright (C) 2009 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University, 2009-2012
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>


#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <fstream>
#include <sstream>

#include "bmop_common.h"

using namespace dealii;




template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem (unsigned int deg);
  void run (int nref);

private:
  std::size_t count_nonzeros() const;

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;

  int degree;
};



template <int dim>
LaplaceProblem<dim>::LaplaceProblem (unsigned int deg)
  :
  fe (deg),
  dof_handler (triangulation),
  degree(deg)
{}

template <int dim>
std::size_t LaplaceProblem<dim>::count_nonzeros() const
{
  ConstraintMatrix  constraints;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);
  DoFTools::make_hanging_node_constraints(dof_handler,constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  return sparsity_pattern.n_nonzero_elements ();
}

template <int dim>
void LaplaceProblem<dim>::run (int n_ref)
{

  domain_case_t domain = BALL;
  // domain_case_t domain = CUBE;

  // bool pseudo_adaptive_grid = true;
  bool pseudo_adaptive_grid = false;

  bmop_setup_mesh(triangulation, domain,
                  pseudo_adaptive_grid, n_ref);


  dof_handler.distribute_dofs (fe);

  // printf("%d\t%d\t%d\t%d\n",dim,degree,
         // triangulation.n_active_cells(),dof_handler.n_dofs());

  printf("%d\t%d\t%d\t%d\t%ld\n",dim,degree,
         triangulation.n_active_cells(),dof_handler.n_dofs(),
         count_nonzeros());

}


int main (int argc, char **argv)
{
  try
  {

    int max_refinement = 1;
    int min_refinement = 0;
    int dim = 2;
    int deg = 3;

    if(argc > 1)
      dim=atoi(argv[1]);

    if(argc > 2)
      deg=atoi(argv[2]);

    if(argc > 3)
      max_refinement = atoi(argv[3]);

    if(argc > 4)
      min_refinement = atoi(argv[4]);

    deallog.depth_console(0);


    for(int r=min_refinement; r<=max_refinement; r++) {
      if(dim==2) {
        LaplaceProblem<2> laplace_problem(deg);
        laplace_problem.run (r );
      }
      else {
        LaplaceProblem<3> laplace_problem(deg);
        laplace_problem.run ( r);
      }
    }
  }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
