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
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <fstream>
#include <sstream>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"

#include "laplace_operator_gpu.h"

using namespace dealii;

typedef double number;

// #define USE_HANGING_NODES

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
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  typedef LaplaceOperatorGpu<dim,fe_degree,number> SystemMatrixType;

  SystemMatrixType                 system_matrix;

  GpuVector<number>                solution;
  GpuVector<number>                system_rhs;

  double                           setup_time;
  ConditionalOStream               time_details;
};



template <int dim, int fe_degree>
LaplaceProblem<dim,fe_degree>::LaplaceProblem ()
  :
  fe (fe_degree),
  dof_handler (triangulation),
  time_details (std::cout, false)
{}




template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::setup_system ()
{
  Timer time;
  time.start ();
  setup_time = 0;

  system_matrix.clear();

  dof_handler.distribute_dofs (fe);

  std::cout << "Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  std::cout << "Number of elements: "
            << dof_handler.get_tria().n_active_cells()
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

  system_matrix.reinit (dof_handler, constraints);

  std::cout.precision(4);
  std::cout << "System matrix memory consumption:     "
            << system_matrix.memory_consumption()*1e-6
            << " MB."
            << std::endl;

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_system ()
{
  Timer time;
  QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values );

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Vector<number> local_diagonal(dofs_per_cell);

  std::vector<number> coefficient_values(n_q_points);

  Vector<number> diagonal(dof_handler.n_dofs());

  Vector<number> system_rhs_host(dof_handler.n_dofs());

  Coeff<dim> coeff;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    fe_values.reinit (cell);

    // coefficient needed here for diagonal
    coeff.value_list(fe_values.get_quadrature_points(), coefficient_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {


      number rhs_val = 0;
      number local_diag = 0;
      for (unsigned int q=0; q<n_q_points; ++q) {
        rhs_val += (fe_values.shape_value(i,q) * 1.0 *
                    fe_values.JxW(q));


        local_diag += ((fe_values.shape_grad(i,q) *
                        fe_values.shape_grad(i,q)) *
                       coefficient_values[q] * fe_values.JxW(q));
      }
      local_diagonal(i) = local_diag;
      system_rhs_host(local_dof_indices[i]) += rhs_val;
    }

    constraints.distribute_local_to_global(local_diagonal,
                                           local_dof_indices,
                                           diagonal);
  }

  system_matrix.set_diagonal(diagonal);

  constraints.condense(system_rhs_host);

  system_rhs = system_rhs_host;

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  typedef PreconditionChebyshev<SystemMatrixType,GpuVector<number> > PreconditionType;

  PreconditionType preconditioner;
  typename PreconditionType::AdditionalData additional_data;

  additional_data.matrix_diagonal_inverse.reinit(system_matrix.m());
  additional_data.matrix_diagonal_inverse = 1.0;
  additional_data.matrix_diagonal_inverse /= system_matrix.get_diagonal();

  preconditioner.initialize(system_matrix,additional_data);

  SolverControl           solver_control (10000, 1e-12*system_rhs.l2_norm());
  SolverCG<GpuVector<number> >              cg (solver_control);
  setup_time += time.wall_time();
  time_details << "Solver/prec. setup time    (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  std::cout << "Total setup time               (wall) " << setup_time
            << "s\n";

  time.reset();
  time.start();
  cg.solve (system_matrix, solution, system_rhs,
            // PreconditionIdentity());
            preconditioner);

  time.stop();
#ifdef FINE_GRAIN_TIMER
  system_matrix.print_fine_grain_timers();
#endif

  std::cout << "Time solve ("
            << solver_control.last_step()
            << " iterations)  (CPU/wall) " << time() << "s/"
            << time.wall_time() << "s\n";
}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::output_results (const unsigned int cycle) const
{
  Vector<number> host_solution = solution.toVector();
  constraints.distribute(host_solution);
  Vector<number> host_numbering = solution.toVector();

  for(int i = 0; i < host_numbering.size(); ++i) {
    host_numbering[i] = i;
  }
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (host_solution, "solution");
  data_out.add_data_vector (host_numbering, "numbering");
  data_out.build_patches (fe_degree);

  std::ostringstream filename;
  filename << "solution-"
           << cycle
           << ".vtu";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run ()
{
  for (unsigned int cycle=0; cycle<7-dim; ++cycle)
  {
    std::cout << "Cycle " << cycle << std::endl;

    if (cycle == 0)
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
          bool ref = true;
          for(int d = 0; d < dim; ++d)
            ref = (p[d] > 0.5) && ref;
          if(ref) it->set_refine_flag();
#else
           it->set_refine_flag();
#endif

        }
        triangulation.execute_coarsening_and_refinement();
      }

    }
    else {
      triangulation.refine_global (1);
    }
    setup_system ();
    assemble_system ();
    solve ();
    output_results (cycle);

    std::cout << std::endl;
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


int main()
{

  printf("d: %d, p: %d\n",dimension,degree_finite_element);

  try
  {
    deallog.depth_console(0);
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
