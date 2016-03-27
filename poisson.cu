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

#include <fstream>
#include <sstream>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/cuda_utils.cuh"

#include "laplace_operator_gpu.h"

using namespace dealii;

typedef double number;

// #define USE_HANGING_NODES

//=============================================================================
// reference solution and right-hand side
//=============================================================================

template <int dim>
class Solution : public Function<dim>
{
private:
  static const unsigned int n_source_centers = 3;
  static const Point<dim>   source_centers[n_source_centers];
  static const double       width;

public:
  Solution () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;

  virtual double laplacian (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
};

template <>
const Point<1>
Solution<1>::source_centers[Solution<1>::n_source_centers]
= { Point<1>(-1.0 / 3.0),
    Point<1>(0.0),
    Point<1>(+1.0 / 3.0)   };

template <>
const Point<2>
Solution<2>::source_centers[Solution<2>::n_source_centers]
= { Point<2>(-0.5, +0.5),
    Point<2>(-0.5, -0.5),
    Point<2>(+0.5, -0.5)   };

template <>
const Point<3>
Solution<3>::source_centers[Solution<3>::n_source_centers]
= { Point<3>(-0.5, +0.5, 0.25),
    Point<3>(-0.6, -0.5, -0.125),
    Point<3>(+0.5, -0.5, 0.5)   };

template <int dim>
const double
Solution<dim>::width = 1./3.;


template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
    return_value += std::exp(-x_minus_xi.norm_square() /
                             (this->width * this->width));
  }

  return return_value /
    Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}



template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  const double pi = numbers::PI;
  Tensor<1,dim> return_value;

  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

    return_value += (-2 / (this->width * this->width) *
                     std::exp(-x_minus_xi.norm_square() /
                              (this->width * this->width)) *
                     x_minus_xi);
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
                                                    this->width);
}

template <int dim>
double Solution<dim>::laplacian (const Point<dim>   &p,
                                 const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

    double laplacian =
      ((-2*dim + 4*x_minus_xi.norm_square()/
        (this->width * this->width)) /
       (this->width * this->width) *
       std::exp(-x_minus_xi.norm_square() /
                (this->width * this->width)));
    return_value += laplacian;
  }
  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
                                                    this->width);
}

// Wrapper for coefficient
template <int dim>
class CoefficientFun : Function<dim>
{
public:
  CoefficientFun () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;

  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};

template <int dim>
double CoefficientFun<dim>::value (const Point<dim>   &p,
                                   const unsigned int) const
{
  return Coefficient<dim>::value(p); // 1. / (0.05 + 2.*((p-x_c).norm_square()));
}

template <int dim>
void CoefficientFun<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double>            &values,
                                      const unsigned int              component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],component);
}


template <int dim>
Tensor<1,dim> CoefficientFun<dim>::gradient (const Point<dim>   &p,
                                             const unsigned int) const
{
  return Coefficient<dim>::gradient(p);
}


// function computing the right-hand side
template <int dim>
class RightHandSide : public Function<dim>
{
private:
  Solution<dim> solution;
  CoefficientFun<dim> coefficient;
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim>   &p,
                                  const unsigned int) const
{
  return -(solution.laplacian(p)*coefficient.value(p)
           + coefficient.gradient(p)*solution.gradient(p));
}

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

  Vector<number> diagonal(dof_handler.n_dofs());
  Vector<number> local_diagonal(dofs_per_cell);

  std::vector<number> coefficient_values(n_q_points);

  Vector<number> system_rhs_host(dof_handler.n_dofs());
  RightHandSide<dim> right_hand_side;
  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<1,dim> > solution_gradients(n_q_points);

  CoefficientFun<dim> coeff;

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell->get_dof_indices (local_dof_indices);
    fe_values.reinit (cell);

    // coefficient needed here for diagonal
    coeff.value_list(fe_values.get_quadrature_points(), coefficient_values);

    fe_values.get_function_gradients(solution_host, solution_gradients);
    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {

      number rhs_val = 0;
      number local_diag = 0;

      for (unsigned int q=0; q<n_q_points; ++q) {
        rhs_val += ((fe_values.shape_value(i,q) * rhs_values[q]
                     - fe_values.shape_grad(i,q) * solution_gradients[q]
                     ) *
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
  cg.solve (system_matrix, solution_update, system_rhs,
            // PreconditionIdentity());
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
