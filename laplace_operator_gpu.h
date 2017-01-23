/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#ifndef _LAPLACE_OPERATOR_GPU_H
#define _LAPLACE_OPERATOR_GPU_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/diagonal_matrix.h>


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_list.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "poisson_common.h"

using namespace dealii;



//=============================================================================
// operator
//=============================================================================

template <int dim, int fe_degree,typename Number>
class LaplaceOperatorGpu : public Subscriptor
{
public:
  typedef Number value_type;
  typedef GpuVector<Number> VectorType;

  LaplaceOperatorGpu ();

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
  Number el (const unsigned int row,
             const unsigned int col) const {
    ExcNotImplemented();
    return -1000000000000000000;
  }

  // diagonal for preconditioning
  void set_diagonal (const Vector<Number> &diagonal);

  const std::shared_ptr<DiagonalMatrix<VectorType>> get_diagonal_inverse () const;

  std::size_t memory_consumption () const;


private:

  void evaluate_coefficient();

  MatrixFreeGpu<dim,Number>   data;
  VectorType                  coefficient;

  std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
  bool                                                diagonal_is_available;

  GpuList<unsigned int>        constrained_indices;
  mutable VectorType           constrained_values_src;
  mutable VectorType           constrained_values_dst;
};



template <int dim, int fe_degree, typename Number>
LaplaceOperatorGpu<dim,fe_degree,Number>::LaplaceOperatorGpu ()
  :
  Subscriptor()
{
  inverse_diagonal_matrix = std::make_shared<DiagonalMatrix<VectorType>>();
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::clear ()
{
  data.free();
  diagonal_is_available = false;
  inverse_diagonal_matrix->clear();
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::reinit (const DoFHandler<dim>  &dof_handler,
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

  // enable hanging nodes
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);


  const int n_constrained_dofs = constraints.n_constraints();

  std::vector<unsigned int> constrained_dofs_host(n_constrained_dofs);

  unsigned int iconstr = 0;
  for(unsigned int i=0; i<dof_handler.n_dofs(); i++) {
    if(constraints.is_constrained(i)) {
      constrained_dofs_host[iconstr] = i;
      iconstr++;
    }
  }

  constrained_indices = constrained_dofs_host;

  constrained_values_dst.reinit(constrained_indices.size());
  constrained_values_src.reinit(constrained_indices.size());

  evaluate_coefficient();
}


//  initialize coefficient

template <int dim, int fe_degree, typename Number>
struct LocalCoeffOp {
  static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

  static __device__ void eval (Number                     *coefficient,
                               const GpuArray<dim,Number> *qpts)
  {

    for (unsigned int q=0; q<n_q_points; ++q) {
      coefficient[q] =  Coefficient<dim>::value(qpts[q]);
    }
  }
};


template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>:: evaluate_coefficient ()
{
  data.template evaluate_on_cells<LocalCoeffOp<dim,fe_degree,Number> >(coefficient);
}


// multiplication/application functions (symmetric operator -> vmult == Tvmult)

template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult (VectorType       &dst,
                                                 const VectorType &src) const
{
  dst = 0.0;
  vmult_add (dst, src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult (VectorType       &dst,
                                                  const VectorType &src) const
{
  dst = 0.0;
  vmult_add (dst,src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult_add (VectorType       &dst,
                                                      const VectorType &src) const
{
  vmult_add (dst,src);
}

// This is the struct we pass to matrix-free for evaluation on each cell

template <int dim, int fe_degree, typename Number>
struct LocalOperator {
  // coefficient values at this cell
  const Number *coefficient;
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
  static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

  // what to do on each quadrature point
  template <typename FEE>
  __device__ inline void quad_operation(FEE *phi, const unsigned int q) const
  {
    phi->submit_gradient (coefficient[phi->get_global_q(q)] * phi->get_gradient(q), q);
  }

  // what to do fore each cell
  __device__ void cell_apply (Number                          *dst,
                              const Number                    *src,
                              const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                              const unsigned int cell,
                              SharedData<dim,Number> *shdata) const
  {
    FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

    phi.read_dof_values(src);

    phi.evaluate (false,true);

    // apply the local operation above
    phi.apply_quad_point_operations(this);

    phi.integrate (false,true);

    phi.distribute_local_to_global (dst);
  }
};



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult_add (VectorType       &dst,
                                                     const VectorType &src) const
{
  // save possibly non-zero values of Dirichlet and hanging-node values on input
  // and output, and set input values to zero to avoid polluting output.
  constrained_values_src = src[constrained_indices];
  const_cast<VectorType&>(src)[constrained_indices] = 0.0;
  constrained_values_dst = const_cast<const VectorType &>(dst)[constrained_indices];
  // constrained_values_dst = dst[constrained_indices];


  // apply laplace operator
  LocalOperator<dim,fe_degree,Number> loc_op;
  loc_op.coefficient = coefficient.getDataRO();

  data.cell_loop (dst,src,loc_op);

  // overwrite Dirichlet values in output with correct values, and reset input
  // to possibly non-zero values.
  dst[constrained_indices] = constrained_values_dst + constrained_values_src;
  const_cast<VectorType&>(src)[constrained_indices] = constrained_values_src;

}

// set diagonal (and set values correponding to constrained DoFs to 1)
template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::set_diagonal(const Vector<Number> &diagonal)
{
  AssertDimension (m(), diagonal.size());

  VectorType &diag = inverse_diagonal_matrix->get_vector();

  diag.reinit(m());
  diag = 1.0;
  diag /= VectorType(diagonal);

  diag[constrained_indices] = 1.0;

  diagonal_is_available = true;
}

template <int dim, int fe_degree, typename Number>
const std::shared_ptr<DiagonalMatrix<GpuVector<Number>>>
LaplaceOperatorGpu<dim,fe_degree,Number>::get_diagonal_inverse() const
{
  Assert (diagonal_is_available == true, ExcNotInitialized());
  return inverse_diagonal_matrix;
}




template <int dim, int fe_degree, typename Number>
std::size_t
LaplaceOperatorGpu<dim,fe_degree,Number>::memory_consumption () const
{
  std::size_t bytes = (data.memory_consumption () +
                       MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
                       MemoryConsumption::memory_consumption(diagonal_is_available) +
                       coefficient.memory_consumption());

  return bytes;
}


#endif /* _LAPLACE_OPERATOR_GPU_H */
