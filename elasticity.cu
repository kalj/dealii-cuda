/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
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
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"


#include <fstream>
#include <iostream>

using namespace dealii;

//---------------------------------------------------------------------------
// Operator
//---------------------------------------------------------------------------


template <int dim, int fe_degree, typename number>
class ElasticityOperatorGpu : public Subscriptor
{
public:
  typedef GpuVector<number> VectorType;

  ElasticityOperatorGpu ();

  void clear();

  void reinit (const DoFHandler<dim>  &dof_handler,
               const ConstraintMatrix  &constraints,
               const unsigned int      level = numbers::invalid_unsigned_int);

  unsigned int m () const { return data.n_dofs; }
  unsigned int n () const { return data.n_dofs; }

  void vmult (VectorType &dst,
              const VectorType &src) const;
  void Tvmult (VectorType &dst,
               const VectorType &src) const;
  void vmult_add (VectorType &dst,
                  const VectorType &src) const;
  void Tvmult_add (VectorType &dst,
                   const VectorType &src) const;

  // we cannot access matrix elements of a matrix free operator directly.
  number el (const unsigned int row,
             const unsigned int col) const {
    ExcNotImplemented();
    return -1000000000000000000;
  }


  // diagonal for preconditioning
  void set_diagonal (const VectorType &diagonal);

  const std::shared_ptr<DiagonalMatrix<VectorType>> get_diagonal_inverse () const;

  std::size_t memory_consumption () const;

private:
  void local_apply (const MatrixFree<dim,number>    &data,
                    VectorType                      &dst,
                    const VectorType                &src,
                    const std::pair<unsigned int,unsigned int> &cell_range) const;

  // void evaluate_coefficient();

  MatrixFreeGpu<dim,number>      data;
  // Table<2, VectorizedArray<number> > coefficient;

  std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
  bool            diagonal_is_available;

  mutable VectorType           temp_dst;
  mutable VectorType           temp_src;
};


template <int dim, int fe_degree, typename number>
ElasticityOperatorGpu<dim,fe_degree,number>::ElasticityOperatorGpu ()
  :
  Subscriptor()
{
  inverse_diagonal_matrix = std::make_shared<DiagonalMatrix<VectorType>>();
}



template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::clear ()
{
  data.clear();
  diagonal_is_available = false;
  inverse_diagonal_matrix->clear();
}

template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                     const ConstraintMatrix  &constraints,
                                                     const unsigned int      level)
{

  typename MatrixFreeGpu<dim,Number>::AdditionalData additional_data;

#ifdef MATRIX_FREE_COLOR
  additional_data.use_coloring = true;
#else
  additional_data.use_coloring = false;
#endif

  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_in_elem;
  additional_data.level_mg_handler = level;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);

  temp_dst.reinit(data.n_constrained_dofs);
  temp_src.reinit(data.n_constrained_dofs);

}



template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::vmult (VectorType       &dst,
                                                    const VectorType &src) const
{
  dst = 0;
  vmult_add (dst, src);
}



template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::Tvmult (VectorType       &dst,
                                                     const VectorType &src) const
{
  dst = 0;
  vmult_add (dst,src);
}



template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::Tvmult_add (VectorType       &dst,
                                                         const VectorType &src) const
{
  vmult_add (dst,src);
}


// This is the struct we pass to matrix-free for evaluation on each cell

template <int dim, int fe_degree, typename Number>
struct LocalOperator {
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
  static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;
  static const unsigned int n_components = dim;

  // what to do on each quadrature point
  template <typename FEE>
  __device__ inline void quad_operation(FEE *phi, const unsigned int q) const
  {
    const Tensor<2,dim,number> grad_u = phi->get_gradient(q);
    const number div_u = trace(grad_u);

    Tensor<2,dim,number> a = mu * (grad_u + transpose(grad_u));

    for(int d = 0; d < dim; ++d)
      a[d][d] += lambda * div_u;

    phi->submit_gradient (a, q);
  }

  // what to do fore each cell
  __device__ void cell_apply (Number                                            *dst,
                              const Number                                      *src,
                              const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                              const unsigned int                                cell,
                              SharedData<dim,n_components,Number>               *shdata) const
  {
    FEEvaluationGpu<dim,fe_degree,n_componentents,Number> phi (cell, gpu_data, shdata);

    phi.read_dof_values(src);

    phi.evaluate (false,true);

    // apply the local operation above
    phi.apply_quad_point_operations(this);

    phi.integrate (false,true);

    phi.distribute_local_to_global (dst);
  }
};




template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::vmult_add (VectorType       &dst,
                                                        const VectorType &src) const
{

  // save possibly non-zero values of Dirichlet values on input and output, and
  // set input values to zero to avoid polluting output.
  data.save_constrained_values(dst, const_cast<VectorType&>(src),
                               temp_src, temp_dst);

  // apply laplace operator
  LocalOperator<dim,fe_degree,Number> loc_op;

  data.cell_loop (dst,src,loc_op);

  // overwrite Dirichlet values in output with correct values, and reset input
  // to possibly non-zero values.
  data.load_constrained_values(dst, const_cast<VectorType&>(src),
                               temp_src, temp_dst);

}


template <int dim, int fe_degree, typename number>
void
ElasticityOperatorGpu<dim,fe_degree,number>::set_diagonal(const VectorType &diagonal)
{
  AssertDimension (m(), diagonal.size());

  GpuVector<number> &diag = inverse_diagonal_matrix->get_vector();

  diag.reinit(m());
  diag = 1.0;
  diag /= GpuVector<number>(diagonal);

  data.set_constrained_values(diag,1.0);

  diagonal_is_available = true;
}

template <int dim, int fe_degree, typename number>
const std::shared_ptr<DiagonalMatrix<typename ElasticityOperatorGpu<dim,fe_degree,number>::VectorType>>
ElasticityOperatorGpu<dim,fe_degree,number>::get_diagonal_inverse() const
{
  Assert (diagonal_is_available == true, ExcNotInitialized());
  return inverse_diagonal_matrix;
}


template <int dim, int fe_degree, typename number>
std::size_t
ElasticityOperatorGpu<dim,fe_degree,number>::memory_consumption () const
{
  return (data.memory_consumption () +
          MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
          MemoryConsumption::memory_consumption(diagonal_is_available));
}



//---------------------------------------------------------------------------
// Problem
//---------------------------------------------------------------------------


template <int dim, int fe_degree>
class ElasticProblem
{
public:
  typedef double number;

  ElasticProblem ();
  ~ElasticProblem ();
  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;

  FESystem<dim>        fe;

  ConstraintMatrix     constraints;

  typedef GpuVector<number>                           VectorType;
  typedef ElasticityOperatorGpu<dim,fe_degree,number> SystemMatrixType;
  SystemMatrixType                                    system_matrix;

  VectorType       solution;
  VectorType       system_rhs;
  Vector<number>   solution_host;

  double               setup_time;
};


template <int dim>
void right_hand_side (const std::vector<Point<dim> > &points,
                      std::vector<Tensor<1, dim> >   &values)
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (dim >= 2, ExcNotImplemented());

  Point<dim> point_1, point_2;
  point_1(0) = 0.5;
  point_2(0) = -0.5;

  for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
  {
    if (((points[point_n]-point_1).norm_square() < 0.2*0.2) ||
        ((points[point_n]-point_2).norm_square() < 0.2*0.2))
      values[point_n][0] = 1.0;
    else
      values[point_n][0] = 0.0;

    if (points[point_n].norm_square() < 0.2*0.2)
      values[point_n][1] = 1.0;
    else
      values[point_n][1] = 0.0;
  }
}



template <int dim, int fe_degree>
ElasticProblem<dim,fe_degree>::ElasticProblem ()
  :
  dof_handler (triangulation),
  fe (FE_Q<dim>(fe_degree), dim)
{}

template <int dim, int fe_degree>
ElasticProblem<dim,fe_degree>::~ElasticProblem ()
{
  dof_handler.clear ();
}


template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::setup_system ()
{
  Timer time;
  time.start ();
  setup_time = 0;
  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of active cells:                "
            << triangulation.n_active_cells()
            << std::endl;

  std::cout << "   Number of degrees of freedom:          "
            << dof_handler.n_dofs()
            << std::endl;


  constraints.clear ();
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dim>(dim),
                                            constraints);
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);
  constraints.close ();
  setup_time += time.wall_time();
  time.restart();


  system_matrix.reinit (dof_handler, constraints);


  std::cout.precision(4);
  std::cout << "   System matrix memory consumption:      "
            << system_matrix.memory_consumption()*1e-6
            << " MB."
            << std::endl;

  solution.reinit (dof_handler.n_dofs());
  solution_host.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  setup_time += time.wall_time();
}


template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::assemble_system ()
{
  Timer time;

  QGauss<dim>  quadrature_formula(fe_degree+1);


  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  number lambda(1.), mu(1.);

  std::vector<Tensor<1, dim> > rhs_values (n_q_points);

  Vector<number> diagonal(dof_handler.n_dofs());
  Vector<number> system_rhs_host(dof_handler.n_dofs());

  typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {

    fe_values.reinit (cell);

    right_hand_side (fe_values.get_quadrature_points(), rhs_values);

    cell->get_dof_indices (local_dof_indices);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
    {
      number rhs_val = 0.0;
      number diag_val = 0.0;

      const unsigned int
        component_i = fe.system_to_component_index(i).first;

      for (unsigned int q_point=0; q_point<n_q_points;
           ++q_point)
      {

        diag_val +=
          (
           lambda * (fe_values.shape_grad(i,q_point)[component_i] * fe_values.shape_grad(i,q_point)[component_i])
           + mu * (fe_values.shape_grad(i,q_point)[component_i] * fe_values.shape_grad(i,q_point)[component_i]
                   + fe_values.shape_grad(i,q_point) * fe_values.shape_grad(i,q_point))
           ) * fe_values.JxW(q_point);

        rhs_val += fe_values.shape_value(i,q_point) *
          rhs_values[q_point][component_i] *
          fe_values.JxW(q_point);
      }

      diagonal(local_dof_indices[i]) += diag_val;
      system_rhs_host(local_dof_indices[i]) += rhs_val;
    }
  }

  constraints.condense (diagonal);

  system_matrix.set_diagonal(diagonal);

  constraints.condense (system_rhs_host);
  system_rhs = system_rhs_host;

  setup_time += time.wall_time();
}



template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::solve ()
{
  Timer time;

  SolverControl           solver_control (2000, 1e-12);
  SolverCG<>              cg (solver_control);

  // VectorType &inv_diag_system_matrix = system_matrix.get_diagonal_inverse()->get_vector();

  // number mydiag;
  // VectorType input(dof_handler.n_dofs());;
  // VectorType output(dof_handler.n_dofs());;
  // for(int i = 0; i < dof_handler.n_dofs(); ++i) {
  //   input=0.0;
  //   output=0.0;
  //   input[i] = 1.0;
  //   system_matrix.vmult(output,input);
  //   mydiag = output[i];

  //   printf("diag=%g, sysdiag=%g (1/:%g)\n",mydiag,1/inv_diag_system_matrix[i],inv_diag_system_matrix[i]);
  // }

  typedef PreconditionChebyshev<SystemMatrixType, VectorType > PreconditionType;
  PreconditionType preconditioner;
  typename PreconditionType::AdditionalData additional_data;
  preconditioner.initialize(system_matrix,additional_data);


  // PreconditionIdentity      preconditioner;

  setup_time += time.wall_time();
  std::cout << "   Total setup time                (wall) " << setup_time
            << "s\n";

  time.reset();
  time.start();
  cg.solve (system_matrix, solution, system_rhs,
            PreconditionIdentity());
  // preconditioner);

  time.stop();

  std::cout << "   Time solve ("
            << solver_control.last_step()
            << " iterations)  (CPU/wall) " << time() << "s/"
            << time.wall_time() << "s\n";

  solution_host = solution.toVector();

  constraints.distribute (solution_host);
}


template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(fe_degree+1),
                                      typename FunctionMap<dim>::type(),
                                      solution_host,
                                      estimated_error_per_cell);

  std::cout << "   Estimated error:                       "
            << estimated_error_per_cell.l2_norm() << std::endl;

  triangulation.refine_global();

  // GridRefinement::refine_and_coarsen_fixed_number (triangulation,
  // estimated_error_per_cell,
  // 0.3, 0.03);

  // triangulation.execute_coarsening_and_refinement ();
}


template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::output_results (const unsigned int cycle) const
{
  std::string filename = "solution-";
  filename += ('0' + cycle);
  Assert (cycle < 10, ExcInternalError());

  filename += ".vtk";
  std::ofstream output (filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);


  std::vector<std::string> solution_names;
  switch (dim)
  {
  case 1:
    solution_names.push_back ("displacement");
    break;
  case 2:
    solution_names.push_back ("x_displacement");
    solution_names.push_back ("y_displacement");
    break;
  case 3:
    solution_names.push_back ("x_displacement");
    solution_names.push_back ("y_displacement");
    solution_names.push_back ("z_displacement");
    break;
  default:
    Assert (false, ExcNotImplemented());
  }

  data_out.add_data_vector (solution_host, solution_names);
  data_out.build_patches (fe_degree);
  data_out.write_vtk (output);
}



template <int dim, int fe_degree>
void ElasticProblem<dim,fe_degree>::run ()
{
  // for (unsigned int cycle=0; cycle<1; ++cycle)
  for (unsigned int cycle=0; cycle<8; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      GridGenerator::hyper_cube (triangulation, -1, 1);
      triangulation.refine_global (2);
    }
    else
      refine_grid ();

    setup_system ();

    assemble_system ();
    solve ();
    output_results (cycle);

    std::cout << std::endl;
  }
}

int main ()
{
  try
  {
    ElasticProblem<2,1> elastic_problem_2d;
    elastic_problem_2d.run ();
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
