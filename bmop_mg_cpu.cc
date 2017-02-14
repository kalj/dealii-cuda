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
#include "bmop_common.h"


using namespace dealii;

typedef double number;
typedef double level_number;
// typedef float level_number;

// typedef LinearAlgebra::distributed::Vector<number> VectorType;
typedef Vector<number> VectorType;


template <typename LAPLACEOPERATOR>
class MGTransferMF : public MGTransferPrebuilt< VectorType >
{
public:
  MGTransferMF(const MGLevelObject<LAPLACEOPERATOR> &laplace,
               const MGConstrainedDoFs &mg_constrained_dofs)
    :
    MGTransferPrebuilt<VectorType >(mg_constrained_dofs),
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
              MGLevelObject<VectorType > &dst,
              const InVector &src) const
  {
    for (unsigned int level=dst.min_level();
         level<=dst.max_level(); ++level)
      laplace_operator[level].initialize_dof_vector(dst[level]);
    MGTransferPrebuilt<VectorType >:: copy_to_mg(mg_dof_handler, dst, src);
  }

private:
  const MGLevelObject<LAPLACEOPERATOR> &laplace_operator;
};


// coarse solver
template<typename MatrixType, typename Number>
class MGCoarseIterative : public MGCoarseGridBase<VectorType >
{
public:
  MGCoarseIterative() {}

  void initialize(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
  }

  virtual void operator() (const unsigned int   level,
                           VectorType &dst,
                           const VectorType &src) const
  {
    ReductionControl solver_control (1e4, 1e-50, 1e-10);
    SolverCG<VectorType > solver_coarse (solver_control);
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
  void run (int nref);

private:
  void setup_system ();
  void assemble_system ();
  void assemble_multigrid ();
  void solve ();

  typedef LaplaceOperatorCpu<dim,fe_degree,number> SystemMatrixType;
  typedef LaplaceOperatorCpu<dim,fe_degree,level_number>  LevelMatrixType;

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  SystemMatrixType                 system_matrix;
  MGLevelObject<LevelMatrixType>   mg_matrices;
  MGConstrainedDoFs                mg_constrained_dofs;

  VectorType                       solution;
  VectorType                       solution_update;
  VectorType                       system_rhs;
};



template <int dim, int fe_degree>
LaplaceProblem<dim,fe_degree>::LaplaceProblem ()
  :
  triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
  fe (fe_degree),
  dof_handler (triangulation)
{}




template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::setup_system ()
{
  system_matrix.clear();
  mg_matrices.clear_elements();

  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);

  constraints.clear();
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(),
                                            constraints);
  DoFTools::make_hanging_node_constraints(dof_handler,constraints);
  constraints.close();

  system_matrix.reinit (dof_handler, constraints);

  solution.reinit (dof_handler.n_dofs());
  solution_update.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

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

}



template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_system ()
{

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

  VectorType     diagonal(dof_handler.n_dofs());
  Vector<number> local_diagonal(dofs_per_cell);
  Vector<number> local_rhs(dofs_per_cell);

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

}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_multigrid ()
{
  const unsigned int n_levels = triangulation.n_levels();

  for (unsigned int level=0; level<n_levels; ++level)
    mg_matrices[level].compute_diagonal();
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

  // XXX: this should be with 'number' precision, but LevelMatrixType is with floats
  MGCoarseIterative<LevelMatrixType,number> mg_coarse;
  mg_coarse.initialize(mg_matrices[0]);

  typedef PreconditionChebyshev<LevelMatrixType,VectorType > SMOOTHER;

  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, VectorType >
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


  mg::Matrix<VectorType > mg_matrix(mg_matrices);

  mg::Matrix<VectorType > mg_interface(mg_interface_matrices);

  Multigrid<VectorType > mg(mg_matrix,
                            mg_coarse,
                            mg_transfer,
                            mg_smoother,
                            mg_smoother);

  mg.set_edge_matrices(mg_interface, mg_interface);

  PreconditionMG<dim, VectorType,
                 MGTransferMF<LevelMatrixType> >
    preconditioner(dof_handler, mg, mg_transfer);


  SolverControl           solver_control (10000, 1e-12*system_rhs.l2_norm());
  SolverCG<VectorType >              cg (solver_control);

  time.reset();
  time.start();
  cg.solve (system_matrix, solution_update, system_rhs,
            preconditioner);

  time.stop();

    // printf("%12d %5d %14.8g\n", dof_handler.n_dofs(),solver_control.last_step(), time.wall_time());
    printf("%8d %8d %12d %5d %14.8g\n", dim,fe_degree,dof_handler.n_dofs(),solver_control.last_step(), time.wall_time());


  constraints.distribute(solution_update);
  solution += solution_update;

}







template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run (int n_ref)
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
  assemble_system ();
  assemble_multigrid ();

  solve ();

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
  try
  {
    deallog.depth_console(0);
    int max_refinement = 1;
    int min_refinement = 0;
    if(argc > 1)
      max_refinement = atoi(argv[1]);

    if(argc > 2)
      min_refinement = atoi(argv[2]);


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

  return 0;
}
