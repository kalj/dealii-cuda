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

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include <fstream>
#include <sstream>

#include "laplace_operator_cpu.h"
#include "poisson_common.h"


using namespace dealii;

typedef double number;
typedef double level_number;
// typedef float level_number;


template <typename Number>
using VectorType = Vector<Number>;
// template <typename Number>
// using VectorType = LinearAlgebra::distributed::Vector<Number>;

bool QUIET=false;

template <typename LAPLACEOPERATOR>
class MGTransferMF : public MGTransferPrebuilt< typename LAPLACEOPERATOR::VectorType >
{
public:
  MGTransferMF(const MGLevelObject<LAPLACEOPERATOR> &laplace,
               const MGConstrainedDoFs &mg_constrained_dofs)
    :
    MGTransferPrebuilt<typename LAPLACEOPERATOR::VectorType >(mg_constrained_dofs),
    laplace_operator (laplace)
  {
  }

  /**
   * Overload copy_to_mg from MGTransferPrebuilt to get the vectors compatible
   * with MatrixFree and bypass the crude vector initialization in
   * MGTransferPrebuilt
   */
  template <int dim, class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof_handler,
              MGLevelObject<typename LAPLACEOPERATOR::VectorType > &dst,
              const InVector &src) const
  {
    for (unsigned int level=dst.min_level();
         level<=dst.max_level(); ++level)
      laplace_operator[level].initialize_dof_vector(dst[level]);
    MGTransferPrebuilt<typename LAPLACEOPERATOR::VectorType >:: copy_to_mg(mg_dof_handler, dst, src);
  }

private:
  const MGLevelObject<LAPLACEOPERATOR> &laplace_operator;
};


// coarse solver
template<typename MatrixType, typename Number>
class MGCoarseIterative : public MGCoarseGridBase<VectorType<Number> >
{
public:
  MGCoarseIterative() {}

  void initialize(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
  }

  virtual void operator() (const unsigned int   level,
                           VectorType<Number> &dst,
                           const VectorType<Number> &src) const
  {
    ReductionControl solver_control (1e4, 1e-50, 1e-10);
    SolverCG<VectorType<Number> > solver_coarse (solver_control);
    solver_coarse.solve (*coarse_matrix, dst, src, PreconditionIdentity());
  }

  const MatrixType *coarse_matrix;
};



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
  void setup_system ();
  void assemble_system ();
  void assemble_multigrid ();
  void solve ();
  void run_tests ();
  void output_results (const unsigned int cycle) const;

  typedef LaplaceOperatorCpu<dim,fe_degree,number> SystemMatrixType;
  typedef LaplaceOperatorCpu<dim,fe_degree,level_number>  LevelMatrixType;

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  SystemMatrixType                 system_matrix;
  MGLevelObject<LevelMatrixType>   mg_matrices;
  MGConstrainedDoFs                mg_constrained_dofs;

  VectorType<number>               solution;
  VectorType<number>               solution_update;
  VectorType<number>               system_rhs;

  double                           setup_time;
  ConditionalOStream               time_details;
};



template <int dim, int fe_degree>
LaplaceProblem<dim,fe_degree>::LaplaceProblem ()
  :
  triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
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
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);

  if(!QUIET) {
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    std::cout << "Number of elements: "
              << triangulation.n_active_cells()
              << std::endl;
  }

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
  if(!QUIET) {
    std::cout.precision(4);
    std::cout << "System matrix memory consumption:     "
              << system_matrix.memory_consumption()*1e-6
              << " MB."
              << std::endl;
  }

  solution.reinit (dof_handler.n_dofs());
  solution_update.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
  time.restart();


  // initialize the matrices for the multigrid method on all levels

  const unsigned int nlevels = triangulation.n_levels();
  mg_matrices.resize(0, nlevels-1);

  mg_constrained_dofs.initialize(dof_handler);
  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

  for (unsigned int level=0; level<nlevels; ++level)
    mg_matrices[level].reinit(dof_handler,
                              mg_constrained_dofs,
                              level);

  setup_time += time.wall_time();
  time_details << "Setup matrix-free levels   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;

}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_system ()
{
  Timer time;

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0, Solution<dim>(), boundary_values);
  for (typename std::map<types::global_dof_index, double>::const_iterator
         it = boundary_values.begin(); it!=boundary_values.end(); ++it)
    solution(it->first) = it->second;

  // compute hanging node values close to the boundary
  ConstraintMatrix hn_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler,hn_constraints);
  hn_constraints.close();

  hn_constraints.distribute(solution);

  QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values );

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  VectorType<number>  diagonal(dof_handler.n_dofs());
  Vector<number>      local_diagonal(dofs_per_cell);
  Vector<number>      local_rhs(dofs_per_cell);

  std::vector<number> coefficient_values(n_q_points);

  RightHandSide<dim> right_hand_side;
  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<1,dim> > solution_gradients(n_q_points);

  CoefficientFun<dim,number> coeff;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    fe_values.reinit (cell);

    // coefficient needed here for diagonal
    coeff.value_list(fe_values.get_quadrature_points(), coefficient_values);

    fe_values.get_function_gradients(solution, solution_gradients);
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

    constraints.distribute_local_to_global(local_rhs,local_dof_indices,
                                           system_rhs);
  }

  system_matrix.compute_diagonal();

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_multigrid ()
{
  Timer time;

  const unsigned int n_levels = triangulation.n_levels();

  for (unsigned int level=0; level<n_levels; ++level)
    mg_matrices[level].compute_diagonal();

  setup_time += time.wall_time();
  time_details << "Assemble MG diagonal       (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;

}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run_tests ()
{
  Timer time;

  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
  mg_interface_matrices.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level=0; level<dof_handler.get_triangulation().n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);

  MGTransferMF<LevelMatrixType > mg_transfer(mg_matrices, mg_constrained_dofs);
  mg_transfer.build_matrices(dof_handler);

  setup_time += time.wall_time();
  time_details << "MG build transfer time     (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();

  // XXX: this should be with 'number' precision, but LevelMatrixType is with floats
  MGCoarseIterative<LevelMatrixType,level_number> mg_coarse;
  mg_coarse.initialize(mg_matrices[0]);

  setup_time += time.wall_time();
  time_details << "MG coarse time             (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();


  typedef PreconditionChebyshev<LevelMatrixType,VectorType<level_number> > SMOOTHER;

  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, VectorType<level_number> >
    mg_smoother;

  MGLevelObject<typename SMOOTHER::AdditionalData> smoother_data;
  smoother_data.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level = 0;
       level<dof_handler.get_triangulation().n_global_levels();
       ++level) {
    smoother_data[level].smoothing_range = 15.;
    smoother_data[level].degree = 5;
    smoother_data[level].eig_cg_n_iterations = 15;
    smoother_data[level].preconditioner = mg_matrices[level].get_diagonal_inverse();
  }

  // temporarily disable deallog for the setup of the preconditioner that
  // involves a CG solver for eigenvalue estimation
  deallog.depth_file(0);
  mg_smoother.initialize(mg_matrices, smoother_data);

  mg::Matrix<VectorType<level_number> > mg_matrix(mg_matrices);

  mg::Matrix<VectorType<level_number> > mg_interface(mg_interface_matrices);

  Multigrid<VectorType<level_number> > mg(mg_matrix,
                            mg_coarse,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother);

  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, VectorType<level_number>,
                 MGTransferMF<LevelMatrixType> >
                 preconditioner(dof_handler, mg, mg_transfer);

  {
    const unsigned int n = system_matrix.m();
    VectorType<level_number> check1(n), check2(n),
      tmp(n), check3(n);
    {
      for (unsigned int i=0; i<n; ++i)
        check1[i] = (double)rand()/RAND_MAX;
    }

    system_matrix.vmult(tmp, check1);
    tmp *= -1.0;
    preconditioner.vmult(check2, tmp);
    check2 += check1;

    SMOOTHER smoother;
    typename SMOOTHER::AdditionalData smoother_data;
    smoother_data.preconditioner = system_matrix.get_diagonal_inverse();
    smoother_data.degree = 15;
    smoother_data.eig_cg_n_iterations = 15;
    smoother_data.smoothing_range = 20;

    smoother.initialize(system_matrix, smoother_data);
    smoother.vmult(check3, tmp);
    check3 += check1;


    DataOut<dim> data_out;
    constraints.distribute(check1);
    constraints.distribute(check2);
    constraints.distribute(check3);
    data_out.add_data_vector (dof_handler, check1 , "initial_field");
    data_out.add_data_vector (dof_handler, check2, "mg_cycle");
    data_out.add_data_vector (dof_handler, check3, "chebyshev");
    // data_out.build_patches (data.mapping, data.fe_degree, DataOut<dim>::curved_inner_cells);
    data_out.build_patches (fe_degree);
    std::ofstream out(("cg_sol_" + Utilities::to_string(n)+ ".vtk").c_str());
    data_out.write_vtk (out);

  }

}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
  mg_interface_matrices.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level=0; level<dof_handler.get_triangulation().n_global_levels(); ++level)
    mg_interface_matrices[level].initialize(mg_matrices[level]);


  MGTransferMF<LevelMatrixType > mg_transfer(mg_matrices, mg_constrained_dofs);
  mg_transfer.build_matrices(dof_handler);

  setup_time += time.wall_time();
  time_details << "MG build transfer time     (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();

  // XXX: this should be with 'number' precision, but LevelMatrixType is with floats
  MGCoarseIterative<LevelMatrixType,level_number> mg_coarse;
  mg_coarse.initialize(mg_matrices[0]);

  setup_time += time.wall_time();
  time_details << "MG coarse time             (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();


  typedef PreconditionChebyshev<LevelMatrixType,VectorType<level_number> > SMOOTHER;

  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, VectorType<level_number> >
    mg_smoother;

  MGLevelObject<typename SMOOTHER::AdditionalData> smoother_data;
  smoother_data.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level = 0;
       level<dof_handler.get_triangulation().n_global_levels();
       ++level) {
    smoother_data[level].smoothing_range = 15.;
    smoother_data[level].degree = 5;
    smoother_data[level].eig_cg_n_iterations = 15;
    smoother_data[level].preconditioner = mg_matrices[level].get_diagonal_inverse();
  }

  // temporarily disable deallog for the setup of the preconditioner that
  // involves a CG solver for eigenvalue estimation
  deallog.depth_file(0);
  mg_smoother.initialize(mg_matrices, smoother_data);


  mg::Matrix<VectorType<level_number> > mg_matrix(mg_matrices);

  mg::Matrix<VectorType<level_number> > mg_interface(mg_interface_matrices);

  Multigrid<VectorType<level_number> > mg(mg_matrix,
                            mg_coarse,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother);

  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, VectorType<level_number>,
                 MGTransferMF<LevelMatrixType> >
    preconditioner(dof_handler, mg, mg_transfer);

  const std::size_t multigrid_memory
    = (mg_matrices.memory_consumption() +
       mg_transfer.memory_consumption());
  if(!QUIET) {
    std::cout << "Multigrid objects memory consumption: "
              << multigrid_memory * 1e-6
              << " MB."
              << std::endl;
  }

  SolverControl           solver_control (10000, 1e-12*system_rhs.l2_norm());
  SolverCG<VectorType<number> >              cg (solver_control);
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
            preconditioner);

  time.stop();

  if(!QUIET) {
    std::cout << "Time solve ("
              << solver_control.last_step()
              << " iterations)  (CPU/wall) " << time() << "s/"
              << time.wall_time() << "s\n";
  }
  else {
    printf("%8d %8d %12d %5d %14.8g\n", dim,fe_degree,dof_handler.n_dofs(),solver_control.last_step(), time.wall_time());
  }


  constraints.distribute(solution_update);
  solution += solution_update;


  if(!QUIET) {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
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
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches (fe_degree);

  std::ostringstream filename;
  filename << "solution-"
           << cycle
           << ".vtu";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);
}


template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run (unsigned min_cycle, unsigned max_cycle)
{
  domain_case_t domain = CUBE;
  grid_case_t initial_grid = UNIFORM;
  grid_case_t grid_refinement = UNIFORM;

  create_mesh(triangulation,domain,initial_grid);

  for (unsigned int cycle=0; cycle<min_cycle; ++cycle)
    refine_mesh(triangulation,grid_refinement);

  for (unsigned int cycle=min_cycle; cycle<=max_cycle; ++cycle)
  {
    if(!QUIET) {
      std::cout << "Cycle " << cycle << std::endl;
    }

    if(cycle!=min_cycle)
      refine_mesh(triangulation,grid_refinement);

    setup_system ();
    assemble_system ();
    assemble_multigrid ();

    // run_tests ();

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


int main(int argc, char **argv)
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

  return 0;
}
