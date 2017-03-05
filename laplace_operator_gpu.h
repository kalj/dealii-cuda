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
#include <deal.II/multigrid/mg_constrained_dofs.h>


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/constraint_handler_gpu.h"

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

  void reinit (const DoFHandler<dim>   &dof_handler,
               const ConstraintMatrix  &constraints);

  void reinit (const DoFHandler<dim>    &dof_handler,
               const MGConstrainedDoFs  &mg_constrained_dofs,
               const unsigned int        level);

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
  void vmult_interface_down(VectorType       &dst,
                            const VectorType &src) const;
  void vmult_interface_up(VectorType       &dst,
                          const VectorType &src) const;

  // we cannot access matrix elements of a matrix free operator directly.
  Number el (const unsigned int row,
             const unsigned int col) const {
    ExcNotImplemented();
    return -1000000000000000000;
  }

  // diagonal for preconditioning
  void compute_diagonal ();

  const std::shared_ptr<DiagonalMatrix<VectorType>> get_diagonal_inverse () const;

  std::size_t memory_consumption () const;


private:
  unsigned int                level;

  void evaluate_coefficient();

  MatrixFreeGpu<dim,Number>   data;
  VectorType                  coefficient;

  std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
  bool                                                diagonal_is_available;

  mutable ConstraintHandlerGpu<Number> constraint_handler;
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
LaplaceOperatorGpu<dim,fe_degree,Number>::reinit (const DoFHandler<dim>   &dof_handler,
                                                  const ConstraintMatrix  &constraints)
{
  this->level = numbers::invalid_unsigned_int;

  // this constructor is used for the non-Multigrid case, i.e. when this matrix
  // is the global system matrix

  typename MatrixFreeGpu<dim,Number>::AdditionalData additional_data;

#ifdef MATRIX_FREE_COLOR
  additional_data.use_coloring = true;
#else
  additional_data.use_coloring = false;
#endif

  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_in_elem;
  additional_data.level_mg_handler = numbers::invalid_unsigned_int;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);

  // enable hanging nodes
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);

  constraint_handler.reinit(constraints,dof_handler.n_dofs());

  evaluate_coefficient();

}


template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::reinit (const DoFHandler<dim>    &dof_handler,
                                                  const MGConstrainedDoFs  &mg_constrained_dofs,
                                                  const unsigned int        level)
{
  this->level = level;

  // Multigrid-case constructor, i.e. this matrix is a level-local matrix

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

  // disable hanging nodes by submitting empty ConstraintMatrix
  data.reinit (dof_handler, ConstraintMatrix(), QGauss<1>(fe_degree+1),
               additional_data);

  // constraints are instead handled here
  constraint_handler.reinit(mg_constrained_dofs,
                            level);

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
      coefficient[q] =  Coefficient<dim,Number>::value(qpts[q]);
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

  // what to do for each cell
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
  constraint_handler.save_constrained_values(dst, const_cast<GpuVector<Number>&>(src));

  // apply laplace operator
  LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};

  data.cell_loop (dst,src,loc_op);

  // overwrite Dirichlet values in output with correct values, and reset input
  // to possibly non-zero values.
  constraint_handler.load_and_add_constrained_values(dst, const_cast<GpuVector<Number>&>(src));
}


template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::
vmult_interface_down(VectorType       &dst,
                     const VectorType &src) const
{
  // set zero Dirichlet values on the refinement edges of the input vector (and
  // remember the src values because we need to reset them at the end).
  // since also the real boundary DoFs should be zeroed out, we do everything at once
  constraint_handler.save_constrained_values(const_cast<VectorType&>(src));

  // use temporary destination, is all zero here
  VectorType tmp_dst(dst.size());

  // apply laplace operator
  LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};
  data.cell_loop (tmp_dst,src,loc_op);

  // now zero out everything except the values at the refinement edges,
  dst = 0.0;
  constraint_handler.copy_edge_values(dst,tmp_dst);

  // and restore the src values
  constraint_handler.load_constrained_values(const_cast<VectorType&>(src));
}

template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::
vmult_interface_up(VectorType       &dst,
                   const VectorType &src) const
{
  // printf("calling vmult_if_up... (level=%d)\n",level);
  dst = 0;

  // only use values at refinement edges
  VectorType src_cpy (src.size());
  constraint_handler.copy_edge_values(src_cpy,src);

  // apply laplace operator
  LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};
  data.cell_loop (dst,src_cpy,loc_op);

  // zero out edge values.
  // since boundary values should also be removed, do both at once.
  constraint_handler.set_constrained_values(dst,0.0);
}


template <int dim, int fe_degree, typename Number>
struct DiagonalLocalOperator {
  // coefficient values at this cell
  const Number *coefficient;
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_q_points_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
  static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;

  // what to do for each cell
  __device__ void cell_apply (Number                          *dst,
                              const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                              const unsigned int cell,
                              SharedData<dim,Number> *shdata) const
  {
    FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

    Number my_diagonal = 0.0;

    const unsigned int tid = (dim==1 ? threadIdx.x%n_q_points_1d :
                              dim==2 ? threadIdx.x%n_q_points_1d + n_q_points_1d*threadIdx.y :
                              threadIdx.x%n_q_points_1d + n_q_points_1d*(threadIdx.y + n_q_points_1d*threadIdx.z));

    for (unsigned int i=0; i<n_local_dofs; ++i)
    {
      // set to unit vector
      phi.submit_dof_value(i==tid?1.0:0.0, tid);

      __syncthreads();
      phi.evaluate (false, true);

      phi.submit_gradient (coefficient[phi.get_global_q(tid)] *
                           phi.get_gradient(tid), tid);
      __syncthreads();
      phi.integrate (false, true);

      if(tid==i)
        my_diagonal = phi.get_value(tid);
    }
    __syncthreads();

    phi.submit_dof_value(my_diagonal, tid);

    phi.distribute_local_to_global (dst);
  }
};



// set diagonal (and set values correponding to constrained DoFs to 1)
template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::compute_diagonal()
{
  VectorType &inv_diag = inverse_diagonal_matrix->get_vector();

  inv_diag.reinit(m());

  DiagonalLocalOperator<dim,fe_degree,Number> diag_loc_op{coefficient.getDataRO()};
  data.cell_loop (inv_diag,diag_loc_op);

  constraint_handler.set_constrained_values(inv_diag,1.0);

  inv_diag.invert();

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
                       MemoryConsumption::memory_consumption(constraint_handler) +
                       MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
                       MemoryConsumption::memory_consumption(diagonal_is_available) +
                       coefficient.memory_consumption());

  return bytes;
}


#endif /* _LAPLACE_OPERATOR_GPU_H */
