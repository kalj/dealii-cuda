/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/vector_memory.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <sstream>


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/gpu_array.cuh"

#include "laplace_operator_gpu.h"

using namespace dealii;

#define N_ITERATIONS 100

#ifdef DEGREE_FE
const unsigned int degree_finite_element = DEGREE_FE;
#else
const unsigned int degree_finite_element = 4;
#endif

#ifdef DIMENSION
const unsigned int dimension = DIMENSION;
#else
const unsigned int dimension = 3;
#endif

typedef double number;

//-------------------------------------------------------------------------
// problem
//-------------------------------------------------------------------------

template <int dim, int fe_degree>
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

  typedef LaplaceOperatorGpu<dim,fe_degree,number> SystemMatrixType;

  SystemMatrixType                 system_matrix;

  GpuVector<number>                src;
  GpuVector<number>                dst;

  ConditionalOStream               time_details;
  unsigned int                     n_iterations;
};



template <int dim, int fe_degree>
LaplaceProblem<dim,fe_degree>::LaplaceProblem ()
  :
  fe (fe_degree),
  dof_handler (triangulation),
  time_details (std::cout, false),
  n_iterations(N_ITERATIONS)
{}




template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::setup_system ()
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



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  // IC
  dst = 0.1;

  for(int i = 0; i < n_iterations; ++i) {
    dst.swap(src);

    system_matrix.vmult(dst,src);
  }

  cudaDeviceSynchronize();

  time.stop();

  printf("%d\t%d\t%d\t%g\n",dim,fe_degree,dof_handler.n_dofs(),time.wall_time() / n_iterations);
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run (int n_ref)
{

#ifdef BALL_GRID
  GridGenerator::hyper_ball (triangulation);
  static const SphericalManifold<dim> boundary;
  triangulation.set_all_manifold_ids_on_boundary(0);
  triangulation.set_manifold (0, boundary);
  bool ball= true;
#else
  GridGenerator::hyper_cube (triangulation, -1., 1.);
  bool ball= false;
#endif


#ifdef ADAPTIVE_GRID
  triangulation.refine_global (std::max(n_ref-2,0));
  {
    float reduction = 0;
    if(ball)
      if(dim==2)
        reduction = 0.021;
      else
        reduction = 0.04;
    else if(dim==3)
        reduction = 0.015;

    // the radii in this refinement are adjusted such that we hit similar number
    // of dofs as the globally refined case on a similar situation

    // refine elements inside disk of radius 0.55
    for (typename Triangulation<dim>::active_cell_iterator
           cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned() &&
          cell->center().norm() < 0.55-reduction)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();

    // refine elements in annulus with radius 0.3 < r < 0.42
    for (typename Triangulation<dim>::active_cell_iterator
           cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
      if (cell->is_locally_owned() &&
          cell->center().norm() > 0.3+reduction &&
          cell->center().norm() < 0.42-reduction)
        cell->set_refine_flag();
    triangulation.execute_coarsening_and_refinement();

    // refine elements in annulus with radius 0.335 < r < 0.41 (or 0.388)
    for (typename Triangulation<dim>::active_cell_iterator
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
             Triangulation<dim>::active_cell_iterator
             cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().distance(offset) > 0.17+reduction &&
            cell->center().distance(offset) < 0.33-reduction)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();

      for (typename
             Triangulation<dim>::active_cell_iterator
             cell=triangulation.begin_active(); cell != triangulation.end(); ++cell)
        if (cell->is_locally_owned() &&
            cell->center().distance(offset) > 0.21+reduction &&
            cell->center().distance(offset) < 0.31-reduction)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }

  }
#else
  triangulation.refine_global (n_ref);
#endif


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
      LaplaceProblem<dimension,degree_finite_element> laplace_problem;

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

  GrowingVectorMemory<GpuVector<number> >::release_unused_memory();

  return 0;
}
