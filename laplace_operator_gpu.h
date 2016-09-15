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


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
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
  LaplaceOperatorGpu ();

  void clear();

  void reinit (const DoFHandler<dim>  &dof_handler,
               const ConstraintMatrix  &constraints);

  unsigned int m () const { return data.n_dofs; }
  unsigned int n () const { return data.n_dofs; }

  void vmult (GpuVector<Number> &dst,
              const GpuVector<Number> &src) const;
  void Tvmult (GpuVector<Number> &dst,
               const GpuVector<Number> &src) const;
  void vmult_add (GpuVector<Number> &dst,
                  const GpuVector<Number> &src) const;
  void Tvmult_add (GpuVector<Number> &dst,
                   const GpuVector<Number> &src) const;

  // we cannot access matrix elements of a matrix free operator directly.
  Number el (const unsigned int row,
             const unsigned int col) const {
    ExcNotImplemented();
    return -1000000000000000000;
  }

  // diagonal for preconditioning
  void set_diagonal (const Vector<Number> &diagonal);
  const GpuVector<Number>& get_diagonal () const {
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_values;
  };

  std::size_t memory_consumption () const;



private:

  void evaluate_coefficient();

  MatrixFreeGpu<dim,Number>   data;
  GpuVector<Number >          coefficient;

  GpuVector<Number>           diagonal_values;
  bool                        diagonal_is_available;


};



template <int dim, int fe_degree, typename Number>
LaplaceOperatorGpu<dim,fe_degree,Number>::LaplaceOperatorGpu ()
  :
  Subscriptor()
{}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::clear ()
{
  data.free();
  diagonal_is_available = false;
  diagonal_values.reinit(0);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                  const ConstraintMatrix  &constraints)
{
  typename MatrixFreeGpu<dim,Number>::AdditionalData additional_data;

  additional_data.use_coloring = true;

  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_in_elem;

  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);

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
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult (GpuVector<Number>       &dst,
                                                 const GpuVector<Number> &src) const
{
  dst = 0.0;
  vmult_add (dst, src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult (GpuVector<Number>       &dst,
                                                  const GpuVector<Number> &src) const
{
  dst = 0.0;
  vmult_add (dst,src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult_add (GpuVector<Number>       &dst,
                                                      const GpuVector<Number> &src) const
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
    FEEvaluationGpu<Number,dim,fe_degree> phi (cell, gpu_data, shdata);

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
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult_add (GpuVector<Number>       &dst,
                                                     const GpuVector<Number> &src) const
{
  LocalOperator<dim,fe_degree,Number> loc_op;
  loc_op.coefficient = coefficient.getDataRO();

  data.cell_loop (dst,src,loc_op);

  data.copy_constrained_values(dst,src);

}

// set diagonal (and set values correponding to constrained DoFs to 1)
template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::set_diagonal(const Vector<Number> &diagonal)
{
  AssertDimension (m(), diagonal.size());

  diagonal_values = diagonal;

  data.set_constrained_values(diagonal_values,1.0);

  diagonal_is_available = true;
}



template <int dim, int fe_degree, typename Number>
std::size_t
LaplaceOperatorGpu<dim,fe_degree,Number>::memory_consumption () const
{
  std::size_t bytes = (data.memory_consumption () +
                       diagonal_values.memory_consumption() +
                       MemoryConsumption::memory_consumption(diagonal_is_available) +
                       coefficient.memory_consumption());

  return bytes;
}


#endif /* _LAPLACE_OPERATOR_GPU_H */
