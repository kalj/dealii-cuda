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
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <sstream>

#include "laplace_operator_cpu.h"
#include "bmop_common.h"

using namespace dealii;

#define N_ITERATIONS 100

#ifdef DEGREE_FE
const unsigned int degree_finite_element = DEGREE_FE;
#else
const unsigned int degree_finite_element = 3;
#endif

#ifdef DIMENSION
const unsigned int dimension = DIMENSION;
#else
const unsigned int dimension = 2;
#endif



template <int dim>
class LaplaceProblem
{
public:
  LaplaceProblem ();
  void run (int nref);

private:
  void setup_system ();
  void solve ();

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  typedef LaplaceOperatorCpu<dim,degree_finite_element,double> SystemMatrixType;

  SystemMatrixType                 system_matrix;

  Vector<double>                   src;
  Vector<double>                   dst;

  unsigned int                     n_iterations;
};



template <int dim>
LaplaceProblem<dim>::LaplaceProblem ()
  :
  fe (degree_finite_element),
  dof_handler (triangulation),
  n_iterations(N_ITERATIONS)
{}




template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
  system_matrix.clear();

  dof_handler.distribute_dofs (fe);


  constraints.clear();
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);
  DoFTools::make_hanging_node_constraints(dof_handler,constraints);
  constraints.close();

  system_matrix.reinit (dof_handler, constraints);

  dst.reinit (system_matrix.n());
  src.reinit (system_matrix.n());

}


template <int dim>
void LaplaceProblem<dim>::solve ()
{
  Timer time;

  // IC
  dst = 0.1;

  for(unsigned int i = 0; i < n_iterations; ++i) {
    dst.swap(src);

    system_matrix.vmult(dst,src);
  }

  time.stop();


  printf("%d\t%d\t%d\t%g\n",dim,degree_finite_element,dof_handler.n_dofs(),time.wall_time() / n_iterations);
}




template <int dim>
void LaplaceProblem<dim>::run (int n_ref)
{

#ifdef BALL_GRID
  domain_case_t domain = BALL;
#else
  domain_case_t domain = CUBE;
#endif

#ifdef ADAPTIVE_GRID
  bool pseudo_adaptive_grid = true;
#else
  bool pseudo_adaptive_grid = false;
#endif

  bmop_setup_mesh(triangulation, domain,
                  pseudo_adaptive_grid, n_ref);

  setup_system ();
  solve ();
}


int main (int argc, char **argv)
{
  try
  {

    int max_refinement = 1;
    int min_refinement = 0;
    if(argc > 1)
      max_refinement = atoi(argv[1]);

    if(argc > 2)
      min_refinement = atoi(argv[2]);

    deallog.depth_console(0);


    for(int r=min_refinement; r<=max_refinement; r++) {
      LaplaceProblem<dimension> laplace_problem;

      laplace_problem.run ( r);
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
