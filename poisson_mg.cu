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
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <fstream>
#include <sstream>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"

#include "laplace_operator_gpu.h"

using namespace dealii;

typedef double number;
typedef double level_number;
// typedef float level_number;



//=============================================================================
// XXX: Copied from parallel_multigrid_mf.cc
//=============================================================================
template<typename MatrixType, typename Number>
class MGCoarseIterative : public MGCoarseGridBase<GpuVector<Number> >
{
public:
  MGCoarseIterative() {}

  void initialize(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
  }

  virtual void operator() (const unsigned int   level,
                           GpuVector<Number> &dst,
                           const GpuVector<Number> &src) const
  {
    ReductionControl solver_control (1e4, 1e-50, 1e-10);
    SolverCG<GpuVector<Number> > solver_coarse (solver_control);
    solver_coarse.solve (*coarse_matrix, dst, src, PreconditionIdentity());
  }

  const MatrixType *coarse_matrix;
};

template <int dim, typename MatrixType>
class MGTransferPrebuiltMF : public MGTransferMatrixFreeGpu<dim, typename MatrixType::value_type>
{
public:
  MGTransferPrebuiltMF(const MGLevelObject<MatrixType> &laplace,
                       const MGConstrainedDoFs &mg_constrained_dofs)
    :
    MGTransferMatrixFreeGpu<dim, typename MatrixType::value_type>(mg_constrained_dofs),
    laplace_operator (laplace)
  {};

  /**
   * Overload copy_to_mg from MGTransferPrebuilt to get the vectors compatible
   * with MatrixFree and bypass the crude initialization in MGTransferPrebuilt
   */
  template <class InVector, int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim> &mg_dof,
              MGLevelObject<GpuVector<typename MatrixType::value_type> > &dst,
              const InVector &src) const
  {
    for (unsigned int level=dst.min_level();
         level<=dst.max_level(); ++level)
      laplace_operator[level].initialize_dof_vector(dst[level]);
    MGLevelGlobalTransfer<GpuVector<typename MatrixType::value_type> >::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<MatrixType> &laplace_operator;
};

//=============================================================================






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
  void assemble_multigrid ();
  void solve ();
  void output_results (const unsigned int cycle) const;

  typedef LaplaceOperatorGpu<dim,fe_degree,number> SystemMatrixType;
  typedef LaplaceOperatorGpu<dim,fe_degree,level_number>  LevelMatrixType;

  Triangulation<dim>               triangulation;
  FE_Q<dim>                        fe;
  DoFHandler<dim>                  dof_handler;
  ConstraintMatrix                 constraints;

  SystemMatrixType                 system_matrix;
  MGLevelObject<LevelMatrixType>   mg_matrices;
  MGLevelObject<ConstraintMatrix>  mg_constraints;

  Vector<number>                   solution_host;
  GpuVector<number>                solution_update;
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
  mg_matrices.clear();
  mg_constraints.clear();

  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);

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

  system_matrix.reinit (dof_handler, constraints);
  std::cout.precision(4);
  std::cout << "System matrix memory consumption:     "
            << system_matrix.memory_consumption()*1e-6
            << " MB."
            << std::endl;

  solution_host.reinit (dof_handler.n_dofs());
  solution_update.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  setup_time += time.wall_time();
  time_details << "Setup matrix-free system   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
  time.restart();


  // initialize the matrices for the multigrid method on all levels

  const unsigned int nlevels = triangulation.n_levels();
  mg_matrices.resize(0, nlevels-1);
  mg_constraints.resize (0, nlevels-1);

  typename FunctionMap<dim>::type dirichlet_boundary;
  ZeroFunction<dim>               homogeneous_dirichlet_bc (1);
  dirichlet_boundary[0] = &homogeneous_dirichlet_bc;

  std::vector<std::set<types::global_dof_index> > boundary_indices(triangulation.n_levels());

  MGTools::make_boundary_list (dof_handler,
                               dirichlet_boundary,
                               boundary_indices);

  for (unsigned int level=0; level<nlevels; ++level)
  {
    std::set<types::global_dof_index>::iterator bc_it = boundary_indices[level].begin();

    for ( ; bc_it != boundary_indices[level].end(); ++bc_it)
      mg_constraints[level].add_line(*bc_it);

    // FIXME: add hanging node constraints!

    mg_constraints[level].close();

    mg_matrices[level].reinit(dof_handler,
                              mg_constraints[level],
                              level);
  }

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
    solution_host(it->first) = it->second;

  QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values );

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  Vector<number> system_rhs_host(dof_handler.n_dofs());
  RightHandSide<dim> right_hand_side;
  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<1,dim> > solution_gradients(n_q_points);

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    fe_values.reinit (cell);

    fe_values.get_function_gradients(solution_host, solution_gradients);
    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {

      number rhs_val = 0;

      for (unsigned int q=0; q<n_q_points; ++q) {
        rhs_val += ((fe_values.shape_value(i,q) * rhs_values[q]
                     - fe_values.shape_grad(i,q) * solution_gradients[q]
                     ) *
                    fe_values.JxW(q));


      }
      system_rhs_host(local_dof_indices[i]) += rhs_val;
    }
  }


  constraints.condense(system_rhs_host);

  system_rhs = system_rhs_host;

  setup_time += time.wall_time();
  time_details << "Assemble right hand side   (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;
}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::assemble_multigrid ()
{
  Timer time;

  QGauss<dim>  quadrature_formula(fe.degree+1);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_gradients  | update_inverse_jacobians |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  const CoefficientFun<dim> coeff;
  std::vector<number>       coefficient_values (n_q_points);
  Vector<number>            local_diagonal (dofs_per_cell);

  const unsigned int n_levels = triangulation.n_levels();
  std::vector<Vector<float> > diagonals (n_levels);
  for (unsigned int level=0; level<n_levels; ++level)
    diagonals[level].reinit (dof_handler.n_dofs(level));

  std::vector<unsigned int> cell_no(triangulation.n_levels());
  typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    const unsigned int level = cell->level();
    cell->get_mg_dof_indices (local_dof_indices);
    fe_values.reinit (cell);
    coefficient.value_list (fe_values.get_quadrature_points(),
                            coefficient_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      double local_diag = 0;
      for (unsigned int q=0; q<n_q_points; ++q)
        local_diag += ((fe_values.shape_grad(i,q) *
                        fe_values.shape_grad(i,q)) *
                       coefficient_values[q] * fe_values.JxW(q));
      local_diagonal(i) = local_diag;
    }
    mg_constraints[level].distribute_local_to_global(local_diagonal,
                                                     local_dof_indices,
                                                     diagonals[level]);

  }

  for (unsigned int level=0; level<n_levels; ++level)
    mg_matrices[level].set_diagonal (diagonals[level]);

  setup_time += time.wall_time();
  time_details << "Assemble MG diagonal       (CPU/wall) "
               << time() << "s/" << time.wall_time() << "s" << std::endl;

}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::solve ()
{
  Timer time;

  MGConstrainedDoFs mg_constrained_dofs;
  ZeroFunction<dim> zero_function;
  typename FunctionMap<dim>::type dirichlet_boundary;
  dirichlet_boundary[0] = &zero_function;
  mg_constrained_dofs.initialize(dof_handler, dirichlet_boundary);


  MGTransferPrebuiltMF<dim, LevelMatrixType > mg_transfer(mg_matrices,mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  setup_time += time.wall_time();
  time_details << "MG build transfer time     (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();

  // XXX: this should be with 'number' precision, but LevelMatrixType is with floats
  MGCoarseIterative<LevelMatrixType,number> mg_coarse;
  mg_coarse.initialize(mg_matrices[0]);

  setup_time += time.wall_time();
  time_details << "MG coarse time             (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  time.restart();


  typedef PreconditionChebyshev<LevelMatrixType,GpuVector<number> > SMOOTHER;

  MGSmootherPrecondition<LevelMatrixType, SMOOTHER, GpuVector<number> >
    mg_smoother;

  MGLevelObject<typename SMOOTHER::AdditionalData> smoother_data;
  smoother_data.resize(0, dof_handler.get_triangulation().n_global_levels()-1);
  for (unsigned int level = 0;
       level<dof_handler.get_triangulation().n_global_levels();
       ++level) {
    smoother_data[level].smoothing_range = 15.;
    smoother_data[level].degree = 5;
    smoother_data[level].eig_cg_n_iterations = 15;
    smoother_data[level].matrix_diagonal_inverse =
      mg_matrices[level].get_matrix_diagonal_inverse();
  }

  // temporarily disable deallog for the setup of the preconditioner that
  // involves a CG solver for eigenvalue estimation
  deallog.depth_file(0);
  mg_smoother.initialize(mg_matrices, smoother_data);


  mg::Matrix<GpuVector<number> > mg_matrix(mg_matrices);

  Multigrid<GpuVector<number> > mg(dof_handler,
                                   mg_matrix,
                                   mg_coarse,
                                   mg_transfer,
                                   mg_smoother,
                                   mg_smoother);

  PreconditionMG<dim, GpuVector<number>,
                 MGTransferMatrixFreeGpu<dim,LevelMatrixType> >
                 preconditioner(dof_handler, mg, mg_transfer);

  const std::size_t multigrid_memory
    = (mg_matrices.memory_consumption() +
       mg_transfer.memory_consumption());
  std::cout << "Multigrid objects memory consumption: "
            << multigrid_memory * 1e-6
            << " MB."
            << std::endl;


  SolverControl           solver_control (10000, 1e-12*system_rhs.l2_norm());
  SolverCG<GpuVector<number> >              cg (solver_control);
  setup_time += time.wall_time();
  time_details << "Solver/prec. setup time    (CPU/wall) " << time()
               << "s/" << time.wall_time() << "s\n";
  std::cout << "Total setup time               (wall) " << setup_time
            << "s\n";

  time.reset();
  time.start();
  cg.solve (system_matrix, solution_update, system_rhs,
            preconditioner);

  time.stop();

  std::cout << "Time solve ("
            << solver_control.last_step()
            << " iterations)  (CPU/wall) " << time() << "s/"
            << time.wall_time() << "s\n";


  Vector<number> solution_update_host = solution_update.toVector();
  constraints.distribute(solution_update_host);
  solution_host += solution_update_host;


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

template <int dim>
bool all_criterion(const Point<dim> &p) {
  return true;
}

template <int dim>
bool octant_criterion(const Point<dim> &p) {
  bool ref = true;
  for(int d=0; d<dim; d++)
    ref = ref && p[d] > 0.2;
  return ref;
}

template <int dim>
bool random_criterion(const Point<dim> &p) {
  double r = (double)rand() / RAND_MAX;
  return r<0.5;
}


template <int dim>
void mark_cells(Triangulation<dim> &triangulation,
                bool (*crit)(const Point<dim> &))
{
  typename Triangulation<dim>::active_cell_iterator
    it = triangulation.begin_active(),
    end = triangulation.end();
  for(; it != end; ++it) {

    if(crit(it->center()))
      it->set_refine_flag();
  }
}

template <int dim, int fe_degree>
void LaplaceProblem<dim,fe_degree>::run ()
{
  enum grid_case_t { uniform, nonuniform, random};

  enum domain_case_t { cube, ball, };

  domain_case_t domain = cube;
  grid_case_t initial_grid = uniform;
  grid_case_t grid_refinement = uniform;

  for (unsigned int cycle=0; cycle<7-dim; ++cycle)
  {
    std::cout << "Cycle " << cycle << std::endl;

    if (cycle == 0)
    {
      if(domain == cube) {
        GridGenerator::subdivided_hyper_cube (triangulation, 2, -1., 1.);
      }
      else if(domain == ball) {
        GridGenerator::hyper_ball (triangulation);
        static const SphericalManifold<dim> boundary;
        triangulation.set_all_manifold_ids_on_boundary(0);
        triangulation.set_manifold (0, boundary);
      }

      triangulation.refine_global (3-dim);

      if(initial_grid == uniform)
        mark_cells(triangulation,all_criterion<dim>);
      else if(initial_grid == nonuniform)
        mark_cells(triangulation,octant_criterion<dim>);
      else if(initial_grid == random)
        mark_cells(triangulation,random_criterion<dim>);

      triangulation.execute_coarsening_and_refinement();

    }
    else {
      if(grid_refinement == uniform)
        mark_cells(triangulation,all_criterion<dim>);
      else if(grid_refinement == nonuniform)
        mark_cells(triangulation,octant_criterion<dim>);
      else if(grid_refinement == random)
        mark_cells(triangulation,random_criterion<dim>);

      triangulation.execute_coarsening_and_refinement();
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

  srand(1);
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
