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
#include "matrix_free_gpu/multi_gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/gpu_partitioner.h"

#include "laplace_operator_gpu.h"
#include "bmop_common.h"

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


#ifdef BMOP_USE_FLOATS
typedef float number;
#else
typedef double number;
#endif

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
  void setup_system (int n_partitions);
  void solve ();

  Triangulation<dim>                    triangulation;
  FE_Q<dim>                             fe;
  DoFHandler<dim>                       dof_handler;
  std::shared_ptr<const GpuPartitioner> partitioner;
  std::vector<ConstraintMatrix>         partition_constraints;
  ConstraintMatrix                      global_constraints;

  typedef LaplaceOperatorGpu<dim,fe_degree,number> SystemMatrixType;

  SystemMatrixType                 system_matrix;

  MultiGpuVector<number>                src;
  MultiGpuVector<number>                dst;

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
void LaplaceProblem<dim,fe_degree>::setup_system (int n_partitions)
{
  system_matrix.clear();

  dof_handler.distribute_dofs (fe);

  partitioner.reset(new GpuPartitioner(dof_handler,n_partitions));

  // set up global constraints

  global_constraints.clear();

  DoFTools::make_hanging_node_constraints(dof_handler,
                                          global_constraints);

  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            global_constraints);

  DoFTools::make_hanging_node_constraints(dof_handler,global_constraints);
  global_constraints.close();


  // set up per-partition constraints
  partition_constraints.resize(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {

    IndexSet locally_relevant_dofs;
    partitioner->extract_locally_relevant_dofs (i,locally_relevant_dofs);

    partition_constraints[i].clear();

    partition_constraints[i].reinit(locally_relevant_dofs);

    DoFTools::make_hanging_node_constraints(dof_handler,
                                            partition_constraints[i]);

    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              partition_constraints[i]);

    DoFTools::make_hanging_node_constraints(dof_handler,partition_constraints[i]);
    partition_constraints[i].close();
  }

  system_matrix.reinit (dof_handler, partitioner, partition_constraints);

  dst.reinit (partitioner);
  src.reinit (partitioner);
}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  // IC
  dst = 0.1;

  for(int i = 0; i < 10; ++i) {
    dst.swap(src);

    system_matrix.vmult(dst,src);
  }

  for(int i = 0; i < partitioner->n_partitions(); ++i) {
    cudaSetDevice(partitioner->get_partition_id(i));
    cudaDeviceSynchronize();
  }

  time.restart();


  for(int i = 0; i < n_iterations; ++i) {
    dst.swap(src);

    system_matrix.vmult(dst,src);
  }

  for(int i = 0; i < partitioner->n_partitions(); ++i) {
    cudaSetDevice(partitioner->get_partition_id(i));
    cudaDeviceSynchronize();
  }

  time.stop();

  printf("%d\t%d\t%d\t%g\n",dim,fe_degree,dof_handler.n_dofs(),time.wall_time() / n_iterations);
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run (int n_ref)
{
#ifdef BMOP_N_GPUS
  int n_partitions = BMOP_N_GPUS;
#else
  int n_partitions = 1;
#endif

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


  setup_system (n_partitions);
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

  GrowingVectorMemory<MultiGpuVector<number> >::release_unused_memory();

  return 0;
}
