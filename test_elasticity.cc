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
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

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

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <iostream>


#define LAMBDA 1.0
#define MU 1.0

namespace Step8
{

  using namespace dealii;

  template <int dim, int fe_degree, typename number>
  class ElasticityOperator : public Subscriptor
  {
  public:
    typedef Vector<number> VectorType;

    ElasticityOperator ();

    void clear();

    void reinit (const DoFHandler<dim>  &dof_handler,
                 const ConstraintMatrix  &constraints,
                 const unsigned int      level = numbers::invalid_unsigned_int);

    unsigned int m () const;
    unsigned int n () const;

    void vmult (VectorType &dst,
                const VectorType &src) const;
    void Tvmult (VectorType &dst,
                 const VectorType &src) const;
    void vmult_add (VectorType &dst,
                    const VectorType &src) const;
    void Tvmult_add (VectorType &dst,
                     const VectorType &src) const;

    number el (const unsigned int row,
               const unsigned int col) const;


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

    MatrixFree<dim,number>      data;
    // Table<2, VectorizedArray<number> > coefficient;

    std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
    bool            diagonal_is_available;

  };


  template <int dim, int fe_degree, typename number>
  ElasticityOperator<dim,fe_degree,number>::ElasticityOperator ()
    :
    Subscriptor()
  {
    inverse_diagonal_matrix = std::make_shared<DiagonalMatrix<VectorType>>();
  }



  template <int dim, int fe_degree, typename number>
  unsigned int
  ElasticityOperator<dim,fe_degree,number>::m () const
  {
    return data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, typename number>
  unsigned int
  ElasticityOperator<dim,fe_degree,number>::n () const
  {
    return data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::clear ()
  {
    data.clear();
    diagonal_is_available = false;
    inverse_diagonal_matrix->clear();
  }

  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                    const ConstraintMatrix  &constraints,
                                                    const unsigned int      level)
  {
    typename MatrixFree<dim,number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,number>::AdditionalData::partition_color;
    additional_data.level_mg_handler = level;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points);
    data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
                 additional_data);
    // evaluate_coefficient();
  }


  // template <int dim, int fe_degree, typename number>
  // void
  // ElasticityOperator<dim,fe_degree,number>::
  // evaluate_coefficient ()
  // {
  //   const unsigned int n_cells = data.n_macro_cells();
  //   FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
  //   coefficient.reinit (n_cells, phi.n_q_points);
  //   for (unsigned int cell=0; cell<n_cells; ++cell)
  //   {
  //     phi.reinit (cell);
  //     for (unsigned int q=0; q<phi.n_q_points; ++q)
  //       coefficient(cell,q) =
  //         Coefficient<dim>::value(phi.quadrature_point(q));
  //   }
  // }



  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::
  local_apply (const MatrixFree<dim,number>               &data,
               VectorType                                 &dst,
               const VectorType                           &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {

    const number lambda = LAMBDA;
    const number mu = MU;

    FEEvaluation<dim,fe_degree,fe_degree+1,dim,number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);
      phi.evaluate (false,true,false);

      for (unsigned int q=0; q<phi.n_q_points; ++q) {

        // DOES NOT WORK:

        // const Tensor<2,dim,VectorizedArray<number>> grad_u = phi.get_gradient(q);
        // phi.submit_divergence(lambda * trace(grad_u),q);
        // phi.submit_gradient (mu * (grad_u + transpose(grad_u)),q);

        // same thing:
        // const SymmetricTensor<2,dim,VectorizedArray<number>> sym_grad_u = phi.get_symmetric_gradient(q);
        // const VectorizedArray<number> div = phi.get_divergence(q);

        // phi.submit_symmetric_gradient( make_vectorized_array(2.0*mu) * sym_grad_u,q);
        // phi.submit_divergence(lambda * div, q);


        // THESE WORK:

        // const Tensor<2,dim,VectorizedArray<number>> grad_u = phi.get_gradient(q);
        // const VectorizedArray<number> div_u = trace(grad_u);

        // Tensor<2,dim,VectorizedArray<number>> a = mu * (grad_u + transpose(grad_u));

        // for(int d = 0; d < dim; ++d)
        //   a[d][d] += lambda * div_u;

        // phi.submit_gradient (a, q);

        // same thing:
        typedef VectorizedArray<number>      vec_t;
        typedef SymmetricTensor<2,dim,vec_t> symt_t;

        const symt_t sym_grad_u = phi.get_symmetric_gradient(q);
        const vec_t  div        = phi.get_divergence(q);
        const symt_t eye        = unit_symmetric_tensor<dim,vec_t>();

        phi.submit_symmetric_gradient( make_vectorized_array(2.0*mu) * sym_grad_u
                                       + lambda*div* eye,
                                       q);

      }

      phi.integrate (false,true);
      phi.distribute_local_to_global (dst);
    }
  }



  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::vmult (VectorType       &dst,
                                                   const VectorType &src) const
  {
    dst = 0;
    vmult_add (dst, src);
  }



  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::Tvmult (VectorType       &dst,
                                                    const VectorType &src) const
  {
    dst = 0;
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::Tvmult_add (VectorType       &dst,
                                                        const VectorType &src) const
  {
    vmult_add (dst,src);
  }


  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::vmult_add (VectorType       &dst,
                                                       const VectorType &src) const
  {
    data.cell_loop (&ElasticityOperator::local_apply, this, dst, src);

    const std::vector<unsigned int> &
      constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i) {
      dst(constrained_dofs[i]) += src(constrained_dofs[i]);
      // dst.local_element(constrained_dofs[i]) += src.local_element(constrained_dofs[i]);
    }
  }



  template <int dim, int fe_degree, typename number>
  number
  ElasticityOperator<dim,fe_degree,number>::el (const unsigned int row,
                                                const unsigned int col) const
  {
    Assert (row == col, ExcNotImplemented());
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return 1.0/inverse_diagonal_matrix->operator()(row,col);
  }


  template <int dim, int fe_degree, typename number>
  void
  ElasticityOperator<dim,fe_degree,number>::set_diagonal(const VectorType &diagonal)
  {
    AssertDimension (m(), diagonal.size());

    VectorType &diag = inverse_diagonal_matrix->get_vector();

    diag.reinit(m());
    for(int i = 0; i < m(); ++i) {
      diag[i] = 1.0/diagonal[i];
    }

    const std::vector<unsigned int> &
      constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      diag(constrained_dofs[i]) = 1.0;

    diagonal_is_available = true;
  }

  template <int dim, int fe_degree, typename number>
  const std::shared_ptr<DiagonalMatrix<typename ElasticityOperator<dim,fe_degree,number>::VectorType>>
  ElasticityOperator<dim,fe_degree,number>::get_diagonal_inverse() const
  {
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return inverse_diagonal_matrix;
  }


  template <int dim, int fe_degree, typename number>
  std::size_t
  ElasticityOperator<dim,fe_degree,number>::memory_consumption () const
  {
    return (data.memory_consumption () +
            // coefficient.memory_consumption() +
            MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
            MemoryConsumption::memory_consumption(diagonal_is_available));
  }



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
    void setup_sparse ();
    void setup_mfree();
    void output_results (const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    DoFHandler<dim>      dof_handler;

    FESystem<dim>        fe;

    ConstraintMatrix     constraints_mfree;
    ConstraintMatrix     constraints_sparse;

    SparsityPattern                 sparsity_pattern;
    typedef ElasticityOperator<dim,fe_degree,number> SystemMatrixType;
    SystemMatrixType                system_matrix_mfree;
    SparseMatrix<number>            system_matrix_sparse;

    Vector<number>       output_mfree;
    Vector<number>       input_mfree;

    Vector<number>       output_sparse;
    Vector<number>       input_sparse;

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

    setup_time += time.wall_time();
  }


  template <int dim, int fe_degree>
  void ElasticProblem<dim,fe_degree>::setup_sparse ()
  {

    output_sparse.reinit (dof_handler.n_dofs());
    input_sparse.reinit (dof_handler.n_dofs());


    constraints_sparse.clear ();
    // DoFTools::make_hanging_node_constraints (dof_handler,
                                             // constraints_sparse);
    constraints_sparse.close ();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints_sparse,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from (dsp);

    system_matrix_sparse.reinit (sparsity_pattern);

    QGauss<dim>  quadrature_formula(fe_degree+1);

    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const FEValuesExtractors::Vector displ(0);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<double>     lambda_values (n_q_points);
    std::vector<double>     mu_values (n_q_points);

    ConstantFunction<dim> lambda(LAMBDA), mu(MU);

    std::vector<Tensor<1, dim> > rhs_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        cell_matrix = 0;

        fe_values.reinit (cell);

        lambda.value_list (fe_values.get_quadrature_points(), lambda_values);
        mu.value_list     (fe_values.get_quadrature_points(), mu_values);
        right_hand_side (fe_values.get_quadrature_points(), rhs_values);


        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {

            for (unsigned int j=0; j<dofs_per_cell; ++j)
              {

                for (unsigned int q_point=0; q_point<n_q_points;
                     ++q_point)
                  {

                    const Tensor<2,dim> grad_phi_i     = fe_values[displ].gradient (i, q_point);
                    const double        div_phi_i = fe_values[displ].divergence (i, q_point);

                    const Tensor<2,dim> grad_phi_j = fe_values[displ].gradient (j, q_point);
                    const double        div_phi_j = fe_values[displ].divergence (j, q_point);

                    cell_matrix(i,j) += (lambda_values[q_point] * div_phi_j * div_phi_i
                                         + mu_values[q_point] *
                                         (
                                          double_contract<0,0,1,1>(grad_phi_i,grad_phi_j)
                                          + double_contract<0,0,1,1>(grad_phi_i,transpose(grad_phi_j))
                                          )
                                         ) *fe_values.JxW(q_point);

                  }
              }
          }

        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            const unsigned int
            component_i = fe.system_to_component_index(i).first;

            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              cell_rhs(i) += fe_values.shape_value(i,q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point);
          }

        cell->get_dof_indices (local_dof_indices);
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              system_matrix_sparse.add (local_dof_indices[i],
                                        local_dof_indices[j],
                                        cell_matrix(i,j));

            input_sparse(local_dof_indices[i]) += cell_rhs(i);
          }
      }

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(dim),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix_sparse,
                                        output_sparse,
                                        input_sparse);
  }


  template <int dim, int fe_degree>
  void ElasticProblem<dim,fe_degree>::setup_mfree ()
  {
    Timer time;

    output_mfree.reinit (dof_handler.n_dofs());
    input_mfree.reinit (dof_handler.n_dofs());

    constraints_mfree.clear ();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(dim),
                                              constraints_mfree);
    // DoFTools::make_hanging_node_constraints (dof_handler,
                                             // constraints_mfree);
    constraints_mfree.close ();
    setup_time += time.wall_time();
    time.restart();


    system_matrix_mfree.reinit (dof_handler, constraints_mfree);


    QGauss<dim>  quadrature_formula(fe_degree+1);


    FEValues<dim> fe_values (fe, quadrature_formula,
                             update_values   | update_gradients |
                             update_quadrature_points | update_JxW_values);

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    std::vector<Tensor<1, dim> > rhs_values (n_q_points);

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

        const unsigned int
          component_i = fe.system_to_component_index(i).first;

        for (unsigned int q_point=0; q_point<n_q_points;
             ++q_point)
        {

          rhs_val += fe_values.shape_value(i,q_point) *
            rhs_values[q_point][component_i] *
            fe_values.JxW(q_point);
        }

        input_mfree(local_dof_indices[i]) += rhs_val;
      }
    }

    // constraints_mfree.condense (input_mfree);

    setup_time += time.wall_time();
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

    data_out.add_data_vector (output_mfree, solution_names);

    // for(auto &a : solution_names)
    // {
      // a += "_sparse";
    // }
    // data_out.add_data_vector (output_sparse, solution_names);
    data_out.build_patches (fe_degree);
    data_out.write_vtk (output);
  }



  template <int dim, int fe_degree>
  void ElasticProblem<dim,fe_degree>::run ()
  {

    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (2);

    setup_system ();

    setup_mfree ();

    setup_sparse ();


    input_mfree = 0.0;

    input_mfree[16] = 0.1;
    input_mfree[17] = 0.1;

    input_sparse = input_mfree;

    output_sparse = 0.0;
    output_mfree = 0.0;
    system_matrix_sparse.vmult(output_sparse,input_sparse);
    constraints_sparse.distribute (output_sparse);

    system_matrix_mfree.vmult(output_mfree,input_mfree);
    constraints_mfree.distribute (output_mfree);



    output_mfree -= output_sparse;

    output_results (0);

    std::cout << "norm of diff: " << output_mfree.l2_norm()
              << std::endl;

    std::cout << std::endl;
  }
}

int main ()
{
  try
  {
    Step8::ElasticProblem<2,1> elastic_problem_2d;
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
