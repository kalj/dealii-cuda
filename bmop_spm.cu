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
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <sstream>


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/cuda_sparse_matrix.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "poisson_common.h"

using namespace dealii;

// #define USE_HANGING_NODES
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
  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve ();

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  SparsityPattern                    sparsity_pattern;
  typedef CUDAWrappers::SparseMatrix<number> SystemMatrixType;
  SystemMatrixType                   system_matrix;

  GpuVector<number>                src;
  GpuVector<number>                dst;

  double                           setup_time;
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
  Timer time;
  time.start ();
  setup_time = 0;

  dof_handler.distribute_dofs (fe);

  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  std::cout << "Number of elements: "
            << dof_handler.get_triangulation().n_active_cells()
            << std::endl;

  constraints.clear();
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);
  DoFTools::make_hanging_node_constraints(dof_handler,constraints);
  constraints.close();
  setup_time += time.wall_time();
  time_details << "Distribute DoFs & B.C.     (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
  time.restart();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  dst.reinit (dof_handler.n_dofs());
  src.reinit (dof_handler.n_dofs());

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_system ()
{
  Timer time;

  // assemble matrix
  SparseMatrix<number> system_matrix_host(sparsity_pattern);

  CoefficientFun<dim> coeff;

  const QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values );

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<number>   cell_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  int ncells =  dof_handler.get_triangulation().n_active_cells();

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {


    cell_matrix = 0;

    fe_values.reinit (cell);

    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      const number current_coefficient = coeff.value(fe_values.quadrature_point (q_index));

      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell_matrix(i,j) += (current_coefficient *
                               fe_values.shape_grad(i,q_index) *
                               fe_values.shape_grad(j,q_index) *
                               fe_values.JxW(q_index));
      }
    }

    cell->get_dof_indices (local_dof_indices);
    constraints.distribute_local_to_global (cell_matrix,
                                            local_dof_indices,
                                            system_matrix_host);
  }


  system_matrix.reinit(system_matrix_host);

  setup_time += time.wall_time();
  time_details << "Assemble system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;

  std::cout.precision(4);
  std::cout << "System matrix memory consumption:     "
            << system_matrix.memory_consumption()*1e-6
            << " MB."
            << std::endl;
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


  std::cout << "Time solve ("
            << n_iterations
            << " iterations)  (CPU/wall) " << time() << "s/"
            << time.wall_time() << "s\n";

  std::cout << "Per iteration "
            << time.wall_time() / n_iterations << "s\n";
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run ()
{
  GridGenerator::hyper_cube (triangulation, 0., 1.);

  triangulation.refine_global (1);
  {
    typename Triangulation<dim>::active_cell_iterator
      it = triangulation.begin_active(),
      end = triangulation.end();
    for(; it != end; ++it) {
      Point<dim> p = it->center();

#ifdef USE_HANGING_NODES
          // if(p[0] > 0.5) it->set_refine_flag();
          bool ref = true;
          for(int d = 0; d < dim; ++d)
            ref = (p[d] > 0.5) && ref;
          if(ref) it->set_refine_flag();
#else
           it->set_refine_flag();
#endif

    }
  }

  triangulation.execute_coarsening_and_refinement();
  triangulation.refine_global (2);

  // set up roughly similar grids for different fe_degree (and scale up 2D
  // problem somewhat)
  if(dim == 2) {
    triangulation.refine_global (2);
  }
  else if(dim == 3) {

  }

  if(degree_finite_element ==1) {
    triangulation.refine_global (3);
  }
  else if(degree_finite_element ==2) {
    triangulation.refine_global (2);
  }
  else if(degree_finite_element ==3) {
    triangulation.refine_global (1);
  }
  else if(degree_finite_element ==4) {
    triangulation.refine_global (1);

  }


  setup_system ();
  assemble_system ();
  solve ();
  std::cout << std::endl;
}



int main ()
{
  try
  {
    deallog.depth_console(0);
    printf("d: %d, p: %d\n",dimension,degree_finite_element);
    LaplaceProblem<dimension,degree_finite_element> laplace_problem;
    laplace_problem.run ();
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
