/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <sstream>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/multi_gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"

#include "laplace_operator_gpu.h"

bool QUIET=false;

using namespace dealii;

typedef double number;

//-------------------------------------------------------------------------
// problem
//-------------------------------------------------------------------------


template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  LaplaceProblem ();
  void run (unsigned int min_cycle, unsigned int max_cycle);

private:
  void setup_system (unsigned int n_partitions);
  void assemble_system ();
  void solve ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>                    triangulation;
  FE_Q<dim>                             fe;
  DoFHandler<dim>                       dof_handler;
  std::shared_ptr<GpuPartitioner>       partitioner;
  std::vector<ConstraintMatrix>         partition_constraints;
  ConstraintMatrix                      global_constraints;

  typedef MultiGpuVector<number>                         VectorType;
  // typedef GpuVector<number>                           VectorType;
  typedef LaplaceOperatorGpu<dim,fe_degree,number> SystemMatrixType;

  SystemMatrixType                 system_matrix;

  Vector<number>                   solution_host;
  VectorType                       solution_update;
  VectorType                       system_rhs;

  double                           setup_time;
  ConditionalOStream               time_details;
};



template <int dim, int fe_degree>
LaplaceProblem<dim,fe_degree>::LaplaceProblem ()
  :
  fe (fe_degree),
  dof_handler (triangulation),
  time_details (std::cout, false),
  partitioner(new GpuPartitioner)
{}




template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::setup_system (unsigned int n_partitions)
{
  Timer time;
  time.start ();
  setup_time = 0;

  system_matrix.clear();

  dof_handler.distribute_dofs (fe);

  partitioner->reinit(dof_handler,n_partitions);

  if(!QUIET) {
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    std::cout << "Number of elements: "
              << triangulation.n_active_cells()
              << std::endl;
  }

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

  setup_time += time.wall_time();
  time_details << "Distribute DoFs & B.C.     (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
  time.restart();

  system_matrix.reinit (dof_handler, partitioner, partition_constraints);

  if(!QUIET) {
    std::cout.precision(4);
    std::cout << "System matrix memory consumption:     "
              << system_matrix.memory_consumption()*1e-6
              << " MB."
              << std::endl;
  }

  solution_host.reinit (dof_handler.n_dofs());
  solution_update.reinit (partitioner);
  system_rhs.reinit (partitioner);

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_system ()
{
  Timer time;

  solution_host = 0;

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0, Solution<dim>(), boundary_values);
  for (typename std::map<types::global_dof_index, double>::const_iterator
         it = boundary_values.begin(); it!=boundary_values.end(); ++it)
    solution_host(it->first) = it->second;

  // compute hanging node values close to the boundary
  ConstraintMatrix hn_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler,hn_constraints);
  hn_constraints.close();

  hn_constraints.distribute(solution_host);

  QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values );

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Vector<number> local_rhs(dofs_per_cell);

  std::vector<number> coefficient_values(n_q_points);

  Vector<number> system_rhs_host(dof_handler.n_dofs());
  RightHandSide<dim> right_hand_side;
  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<1,dim,number> > solution_gradients(n_q_points);

  CoefficientFun<dim,number> coeff;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    fe_values.reinit (cell);

    // coefficient needed here for rhs
    coeff.value_list(fe_values.get_quadrature_points(), coefficient_values);

    fe_values.get_function_gradients(solution_host, solution_gradients);
    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      number rhs_val = 0;

      for (unsigned int q=0; q<n_q_points; ++q) {
        rhs_val += ((fe_values.shape_value(i,q) * rhs_values[q]
                     - fe_values.shape_grad(i,q) * solution_gradients[q]
                     * coefficient_values[q]
                     ) *
                    fe_values.JxW(q));
      }

      local_rhs(i) = rhs_val;
    }

    global_constraints.distribute_local_to_global(local_rhs,local_dof_indices,
                                                  system_rhs_host);
  }

  system_matrix.compute_diagonal();
  system_rhs = system_rhs_host;
  system_rhs.update_ghost_values();

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  // typedef PreconditionChebyshev<SystemMatrixType,VectorType > PreconditionType;

  // PreconditionType preconditioner;
  // typename PreconditionType::AdditionalData additional_data;

  // additional_data.preconditioner=system_matrix.get_diagonal_inverse();

  // preconditioner.initialize(system_matrix,additional_data);


  SolverControl           solver_control (10000, 1e-12*system_rhs.l2_norm());
  SolverCG<VectorType >              cg (solver_control);
  setup_time += time.wall_time();
  time_details << "Solver/prec. setup time    (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  if(!QUIET) {
    std::cout << "Total setup time               (wall) " << setup_time
              << "s\n";
  }

  time.reset();
  time.start();
  cg.solve (system_matrix, solution_update, system_rhs,
            // preconditioner);
            PreconditionIdentity());

  time.stop();

  if(!QUIET) {
    std::cout << "Time solve ("
              << solver_control.last_step()
              << " iterations)  (CPU/wall) " << time() << "s/"
              << time.wall_time() << "s\n";
  }
  else {
    printf("%8d %8d %12d %8d %14.8g\n", dim,fe_degree,dof_handler.n_dofs(),
           solver_control.last_step(), time.wall_time());
  }


  Vector<number> solution_update_host = solution_update.toVector();
  global_constraints.distribute(solution_update_host);
  solution_host += solution_update_host;


  if(!QUIET) {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution_host,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm);
    const double L2_error = difference_per_cell.norm_sqr();

    std::cout.precision(6);
    std::cout << "L2 error: " << L2_error << std::endl;
  }

}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::output_results (const unsigned int cycle) const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution_host, "solution");
  data_out.build_patches (fe_degree);

  std::ostringstream filename;
  filename << "solution-"
           << cycle
           << ".vtu";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run (unsigned int min_cycle, unsigned int max_cycle)
{
  domain_case_t domain = CUBE;
  grid_case_t initial_grid = UNIFORM;
  grid_case_t grid_refinement = UNIFORM;
  unsigned int n_partitions = 1;

  create_mesh(triangulation,domain,initial_grid);

  for (unsigned int cycle=0; cycle<min_cycle; ++cycle)
    refine_mesh(triangulation,grid_refinement);

  for (unsigned int cycle=min_cycle; cycle<=max_cycle; ++cycle)
  {
    if(!QUIET) {
      std::cout << "Cycle " << cycle << std::endl;
    }

    if(cycle != min_cycle)
      refine_mesh(triangulation,grid_refinement);

    setup_system (n_partitions);
    assemble_system ();
    solve ();
    if(!QUIET) {
      output_results (cycle);
    }

    if(!QUIET) {
      std::cout << std::endl;
    }

  }

}


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


int main (int argc, char **argv)
{
  unsigned int min_cycle = 0;
  unsigned int max_cycle = 6-dimension;

  srand(1);

  try
  {
    deallog.depth_console(0);
    if(argc > 1)
      if(strcmp("-q",argv[1])==0)
      {
        QUIET=true;
        argv++;
        argc--;
      }

    if(argc > 1)
      min_cycle = atoi(argv[1]);

    if(argc > 2)
      max_cycle = atoi(argv[2]);

    if(!QUIET) printf("d: %d, p: %d\n",dimension,degree_finite_element);

    LaplaceProblem<dimension,degree_finite_element> laplace_problem;
    laplace_problem.run (min_cycle,max_cycle);

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
