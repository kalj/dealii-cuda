/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <sstream>

#include "matrix_free_gpu/gpu_vec.h"
#include "laplace_operator_cpu.h"
#include "laplace_operator_gpu.h"

using namespace dealii;


const unsigned int degree_finite_element = 4;
const unsigned int dimension = 2;

void assemble_matrix (SparseMatrix<double> &system_matrix, Vector<double> &solution,
                      Vector<double> &system_rhs,
                      const FE_Q<dimension> &fe, const DoFHandler<dimension> &dof_handler,
                      const ConstraintMatrix &constraints)
{
  QGauss<dimension>  quadrature_formula(degree_finite_element+1);
  FEValues<dimension> fe_values (fe, quadrature_formula,
                                 update_quadrature_points | update_values | update_gradients | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);

  std::vector<double> coefficient_values(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  CoefficientFun<dimension> coeff;

  DoFHandler<dimension>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    fe_values.reinit (cell);

    // coefficient needed here for rhs (and diagonal)
    coeff.value_list(fe_values.get_quadrature_points(), coefficient_values);

    cell_matrix = 0;

    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
        {
          cell_matrix(i,j) += ((fe_values.shape_grad (i, q_index) *
                                fe_values.shape_grad (j, q_index)) *
                               coefficient_values[q_index] * fe_values.JxW (q_index));

        }
    }
    cell->get_dof_indices (local_dof_indices);

    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i,j));

  }

  for(int i=0; i<system_matrix.m(); i++) {
    if(constraints.is_constrained(i)) {

      for(int j=0; j<system_matrix.m() ; ++j) {
        if(i == j) {
          system_matrix.set(i,i,1.0);
        }
        else {
          system_matrix.set(i,j,0.0);
          system_matrix.set(j,i,0.0);
        }
      }

    }
  }


}



int main (int argc, char **argv)
{
    Triangulation<dimension>               triangulation;
    FE_Q<dimension>                        fe(degree_finite_element);
    DoFHandler<dimension>                  dof_handler(triangulation);
    ConstraintMatrix                 constraints;

    LaplaceOperatorGpu<dimension,degree_finite_element,double> system_matrix_gpu;
    LaplaceOperatorCpu<dimension,degree_finite_element,double> system_matrix_cpu;

    SparseMatrix<double> system_matrix_sparse;

    GridGenerator::hyper_cube (triangulation, 0., 1.);
    triangulation.refine_global (2);

    system_matrix_cpu.clear();
    system_matrix_gpu.clear();
    system_matrix_sparse.clear();

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    constraints.clear();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dimension>(),
                                              constraints);
    constraints.close();

    system_matrix_cpu.reinit (dof_handler, constraints);
    system_matrix_gpu.reinit (dof_handler, constraints);

    SparsityPattern      sparsity_pattern;
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);

    system_matrix_sparse.reinit (sparsity_pattern);


    Vector<double> input(dof_handler.n_dofs());
    Vector<double> res_sparse(dof_handler.n_dofs());

    assemble_matrix(system_matrix_sparse,res_sparse,input,fe,dof_handler,constraints);



    for(int i = 0; i < dof_handler.n_dofs(); ++i) {
      input[i] = (double)rand()/RAND_MAX;
    }

    GpuVector<double> input_gpu(input);

    Vector<double> res_cpu(dof_handler.n_dofs());
    GpuVector<double> res_gpu(dof_handler.n_dofs());

    system_matrix_gpu.vmult(res_gpu,input_gpu);
    system_matrix_cpu.vmult(res_cpu,input);
    system_matrix_sparse.vmult(res_sparse,input);

    Vector<double> res_gpu_host = res_gpu.toVector();

    Vector<double> tmp = res_cpu;
    tmp -= res_gpu_host;
    res_gpu_host -= res_sparse;
    res_cpu -= res_sparse;
    std::cout << "Difference sparse & cpu_mf = " << res_cpu.l2_norm() << std::endl;
    std::cout << "Difference gpu_mf & cpu_mf = " << tmp.l2_norm() << std::endl;
    std::cout << "Difference sparse & gpu_mf = " << res_gpu_host.l2_norm() << std::endl;

    return 0;
}
