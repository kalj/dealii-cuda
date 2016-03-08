/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)fee_gpu.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef __deal2__matrix_free_fee_gpu_h
#define __deal2__matrix_free_fee_gpu_h

#include "defs.h"
#include "utils.h"
#include "tensor_ops.cuh"
#include "matrix_free_gpu.h"
#include "atomic.cuh"

#include "gpu_array.cuh"
#include "hanging_nodes.cuh"

// object which lives on the Gpu which contains methods for element local
// operations, such as gradient and value evaluation. This is created for each
// thread at each multiplication, but since creation only means setting up a few
// pointers.

template <typename Number, int dim, int fe_degree>
class FEEvaluationGpuBase
{
public:
  typedef Number number_type;
  typedef typename MatrixFreeGpu<dim,Number>::GpuData data_type;
  static const unsigned int dimension = dim;
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_q_points_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
  static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;
  typedef GpuArray<dim,Number> gradient_type;

  const unsigned int n_cells;
protected:

  // mapping and mesh info
  const unsigned int *loc2glob;
  const Number       *inv_jac;
  const Number       *JxW;

  // whether coloring is used or not
  const bool       use_coloring;

  const unsigned int cellid;

  const unsigned int constraint_mask;

public:
  const GpuArray<dim,Number> *quadrature_points;

  __device__ FEEvaluationGpuBase(int cellid, const data_type *data)
    :
    n_cells(data->n_cells),
    cellid(cellid),
    constraint_mask(data->constraint_mask[cellid]),
    use_coloring(data->use_coloring)
  {
  }

};


//=============================================================================
// shmem version
//=============================================================================


template <typename Number, int dim, int fe_degree>
class FEEvaluationGpuPIE : public FEEvaluationGpuBase<Number,dim,fe_degree>
{
public:
  typedef Number number_type;
  typedef typename MatrixFreeGpu<dim,Number>::GpuData data_type;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_dofs_1d;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_q_points_1d;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_local_dofs;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_q_points;
  typedef GpuArray<dim,Number> gradient_type;


private:
  using FEEvaluationGpuBase<Number,dim,fe_degree>::loc2glob;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::inv_jac;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::JxW;

  using FEEvaluationGpuBase<Number,dim,fe_degree>::cellid;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_cells;

  using FEEvaluationGpuBase<Number,dim,fe_degree>::use_coloring;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::constraint_mask;


  // internal buffers
  Number             *values;
  Number             *gradients[dim];

public:
  const GpuArray<dim,Number> *quadrature_points;

  __device__ FEEvaluationGpuPIE(int cellid, const data_type *data, SharedData<dim,Number> *shdata);

  __device__ void read_dof_values(const Number *src);
  __device__ void distribute_local_to_global(Number *dst);

  __device__ void evaluate(const bool evaluate_val,
                           const bool evaluate_grad,
                           const bool evaluate_lapl);

  __device__ void submit_value(const Number &val, const unsigned int q);
  __device__ void submit_gradient(const gradient_type &grad, const unsigned int q);

  __device__ Number get_value(const unsigned int q) const ;
  __device__ gradient_type get_gradient(const unsigned int q) const ;

  __device__  const GpuArray<dim,Number> &get_quadrature_point(const unsigned int q) const ;

  __device__ void integrate(const bool integrate_val, const bool integrate_grad);

  template <typename LocOp>
  __device__ void apply_quad_point_operations(const LocOp *lop) {
    const unsigned int q = (threadIdx.x%n_dofs_1d)+n_dofs_1d*threadIdx.y+(dim==3 ?(n_dofs_1d*n_dofs_1d*threadIdx.z) : 0);
    lop->quad_operation(this,q,cellid*n_q_points+q);
  }

};

template <typename Number, int dim, int fe_degree>
__device__ FEEvaluationGpuPIE<Number,dim,fe_degree>::FEEvaluationGpuPIE(int cellid,
                                                                                      const data_type *data,
                                                                                      SharedData<dim,Number> *shdata)
  :
  FEEvaluationGpuBase<Number,dim,fe_degree>(cellid,data)
{
  values = shdata->values;
  loc2glob = data->loc2glob+n_local_dofs*cellid;
  inv_jac = data->inv_jac+n_q_points*cellid;
  JxW = data->JxW+n_q_points*cellid;
  quadrature_points = data->quadrature_points+n_q_points*cellid;

  for(int d = 0; d < dim; ++d) {
    gradients[d] = shdata->gradients[d];
  }
}

template <typename Number, int dim, int fe_degree>
inline
__device__ const GpuArray<dim,Number>& FEEvaluationGpuPIE<Number,dim,fe_degree>::get_quadrature_point(const unsigned int q) const
{
  return quadrature_points[q*n_cells];
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::evaluate(const bool evaluate_val,
                                                                                 const bool evaluate_grad,
                                                                                 const bool evaluate_lapl)
{
  if(evaluate_grad)
    TensorOpsShmem<dim,fe_degree+1,Number>::grad_at_quad_pts (gradients,values);

  if(evaluate_val) {

    if(evaluate_grad) __syncthreads();
    TensorOpsShmem<dim,fe_degree+1,Number>::fun_at_quad_pts (values);
  }
}

template <typename Number, int dim, int fe_degree>
__device__ Number
FEEvaluationGpuPIE<Number,dim,fe_degree>::get_value(const unsigned int q) const
{
  return values[q];
}

template <typename Number, int dim, int fe_degree>
__device__ FEEvaluationGpuPIE<Number,dim,fe_degree>::gradient_type
FEEvaluationGpuPIE<Number,dim,fe_degree>::get_gradient(const unsigned int q) const
{
  // compute J^{-1} * gradients_quad[q]
  gradient_type grad;
  const Number* J = &inv_jac[q];

  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_J0
    grad[d1] = J[0]*gradients[d1][q];
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[n_q_points*n_cells*(dim*d1+d2)] * gradients[d2][q];
    }
    grad[d1] = tmp;
#endif

  }

  return grad;
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::submit_value(const Number &val, const unsigned int q)
{
  const Number jxw = JxW[q];
  values[q] = jxw*val;
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::submit_gradient(const gradient_type &grad, const unsigned int q)
{
  // compute J^{-T} * grad * det(J) *w_q
  const Number *J = &inv_jac[q];
  const Number jxw = JxW[q];

  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_J0
    gradients[d1][q] = grad[d1] * J[0] * jxw;
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[n_cells*n_q_points*(dim*d2+d1)]*grad[d2];
    }
    gradients[d1][q] = tmp * jxw;
#endif

  }
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::integrate(const bool integrate_val,
                                                                                  const bool integrate_grad)
{
  // TODO: merge these when both are called

  if(integrate_val) {
    TensorOpsShmem<dim,fe_degree+1,Number>::quad_int_fun (values);

    if(integrate_grad) {
      __syncthreads();
      TensorOpsShmem<dim,fe_degree+1,Number>::quad_int_grad<true> (values,gradients);
    }
  }
  else if(integrate_grad)
    TensorOpsShmem<dim,fe_degree+1,Number>::quad_int_grad<false> (values,gradients);
}

//=============================================================================
// Read DoF values
//=============================================================================

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::read_dof_values(const Number *src)
{
  const unsigned int  idx = (threadIdx.x%n_q_points_1d)
    +(dim>1 ? threadIdx.y : 0)*n_q_points_1d
    +(dim>2 ? threadIdx.z : 0)*n_q_points_1d*n_q_points_1d;

#ifdef NO_SHARED_DOFS
  const unsigned int srcidx=cellid*n_local_dofs+idx;
#else // NO_SHARED_DOFS
  const unsigned int srcidx = loc2glob[idx];
#endif // NO_SHARED_DOFS
  values[idx] = src[srcidx];

  if(constraint_mask)
    resolve_hanging_nodes_shmem<dim,fe_degree,NOTRANSPOSE>(values,constraint_mask);
}



//=============================================================================
// Distribute local to global
//=============================================================================

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpuPIE<Number,dim,fe_degree>::distribute_local_to_global(Number *dst)
{
  if(constraint_mask)
    resolve_hanging_nodes_shmem<dim,fe_degree,TRANSPOSE>(values,constraint_mask);

#ifdef NO_SHARED_DOFS
  const unsigned int  i = (threadIdx.x%n_q_points_1d)+n_q_points_1d*threadIdx.y+n_q_points_1d*n_q_points_1d*threadIdx.z;

  const unsigned int dstidx = cellid*n_local_dofs+i;
  dst[dstidx] += values[i];
#else // NO_SHARED_DOFS

  if(use_coloring) {

    const unsigned int  i = (threadIdx.x%n_q_points_1d)
      +(dim>1 ? threadIdx.y : 0)*n_q_points_1d
      +(dim>2 ? threadIdx.z : 0)*n_q_points_1d*n_q_points_1d;
    {
      const unsigned int dstidx = loc2glob[i];
      dst[dstidx] += values[i];
    }
  }
  else {
    const unsigned int  i = (threadIdx.x%n_q_points_1d)
      +(dim>1 ? threadIdx.y : 0)*n_q_points_1d
      +(dim>2 ? threadIdx.z : 0)*n_q_points_1d*n_q_points_1d;
    {
      const unsigned int dstidx = loc2glob[i];

#ifdef MATRIX_FREE_NOTHING
      dst[dstidx] += values[i];
#else
      atomicAdd(&dst[dstidx],values[i]);
#endif
    }
  }
#endif // NO_SHARED_DOFS
}


//=============================================================================
// pmem version
//=============================================================================

template <typename Number, int dim, int fe_degree>
class FEEvaluationGpu : public FEEvaluationGpuBase<Number,dim,fe_degree>
{
public:
  typedef Number number_type;
  typedef typename MatrixFreeGpu<dim,Number>::GpuData data_type;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_dofs_1d;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_q_points_1d;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_local_dofs;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_q_points;
  typedef GpuArray<dim,Number> gradient_type;


private:
  using FEEvaluationGpuBase<Number,dim,fe_degree>::loc2glob;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::inv_jac;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::JxW;

  using FEEvaluationGpuBase<Number,dim,fe_degree>::cellid;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::n_cells;

  using FEEvaluationGpuBase<Number,dim,fe_degree>::use_coloring;
  using FEEvaluationGpuBase<Number,dim,fe_degree>::constraint_mask;

  // internal buffers
  Number             values_dofs[n_local_dofs];
  Number             values_quad[n_q_points];
  Number             gradients_quad[dim][n_q_points];

public:
  const GpuArray<dim,Number> *quadrature_points;

  __device__ FEEvaluationGpu(int cellid, const data_type *data);

  __device__ void read_dof_values(const Number *src);
  __device__ void distribute_local_to_global(Number *dst);

  __device__ void evaluate(const bool evaluate_val,
                           const bool evaluate_grad,
                           const bool evaluate_lapl);

  __device__ void submit_value(const Number &val, const unsigned int q);
  __device__ void submit_gradient(const gradient_type &grad, const unsigned int q);

  __device__ Number get_value(const unsigned int q) const ;
  __device__ gradient_type get_gradient(const unsigned int q) const ;

  __device__  const GpuArray<dim,Number> &get_quadrature_point(const unsigned int q) const ;

  __device__ void integrate(const bool integrate_val, const bool integrate_grad);

  template <class LocOp>
  __device__ void apply_quad_point_operations(const LocOp *lop) {
    for (unsigned int q=0; q<n_q_points; ++q) {
      lop->quad_operation(this,q,cellid+n_cells*q);
    }
  }

};

template <typename Number, int dim, int fe_degree>
__device__ FEEvaluationGpu<Number,dim,fe_degree>::FEEvaluationGpu(int cellid,
                                                                                const data_type *data)
  :
  FEEvaluationGpuBase<Number,dim,fe_degree>(cellid,data)
{
  loc2glob = data->loc2glob+cellid;
  inv_jac = data->inv_jac+cellid;
  JxW = data->JxW+cellid;
  quadrature_points = data->quadrature_points+cellid;
}

template <typename Number, int dim, int fe_degree>
inline
__device__ const GpuArray<dim,Number>& FEEvaluationGpu<Number,dim,fe_degree>::get_quadrature_point(const unsigned int q) const
{
  return quadrature_points[q];
}


template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::evaluate(const bool evaluate_val,
                                                                              const bool evaluate_grad,
                                                                              const bool evaluate_lapl)
{
  if(evaluate_grad)
    TensorOpsPmem<dim,fe_degree+1,Number>::grad_at_quad_pts (gradients_quad,&values_dofs[0]);

  if(evaluate_val) {

    TensorOpsPmem<dim,fe_degree+1,Number>::fun_at_quad_pts (&values_quad[0],&values_dofs[0]);
  }
}

template <typename Number, int dim, int fe_degree>
__device__ Number
FEEvaluationGpu<Number,dim,fe_degree>::get_value(const unsigned int q) const
{
  return values_quad[q];
}

template <typename Number, int dim, int fe_degree>
__device__ FEEvaluationGpu<Number,dim,fe_degree>::gradient_type
FEEvaluationGpu<Number,dim,fe_degree>::get_gradient(const unsigned int q) const
{
  // compute J^{-1} * gradients_quad[q]
  gradient_type grad;
  const Number *J = &inv_jac[q*n_cells*dim*dim];

  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_J0
    grad[d1] = J[0]*gradients_quad[d1][q];
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[n_cells*(dim*d1+d2)] * gradients_quad[d2][q];
    }
    grad[d1] = tmp;
#endif

  }

  return grad;
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::submit_value(const Number &val, const unsigned int q)
{
  const Number jxw = JxW[n_cells*q];
  values_quad[q] = jxw*val;
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::submit_gradient(const gradient_type &grad, const unsigned int q)
{
  // compute J^{-T} * grad * det(J) *w_q
  const Number *J = &inv_jac[q*n_cells*dim*dim];
  const Number jxw = JxW[q*n_cells];

  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_J0
    gradients_quad[d1][q] = grad[d1] * J[0] * jxw;
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[n_cells*(dim*d2+d1)]*grad[d2];
    }
    gradients_quad[d1][q] = tmp * jxw;
#endif

  }
}

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::integrate(const bool integrate_val,
                                                                               const bool integrate_grad)
{
  // TODO: merge these when both are called

  if(integrate_val) {
    TensorOpsPmem<dim,fe_degree+1,Number>::quad_int_fun (values_dofs,values_quad);

    if(integrate_grad) {
      TensorOpsPmem<dim,fe_degree+1,Number>::quad_int_grad<true> (values_dofs,gradients_quad);
    }
  }
  else if(integrate_grad)
    TensorOpsPmem<dim,fe_degree+1,Number>::quad_int_grad<false> (values_dofs,gradients_quad);
}

//=============================================================================
// Read DoF values
//=============================================================================

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::read_dof_values(const Number *src)
{
  for(int i=0; i<n_local_dofs; ++i) {
#ifdef NO_SHARED_DOFS
    const unsigned int srcidx = i*n_cells+cellid;
#else // NO_SHARED_DOFS
    const unsigned int srcidx = loc2glob[n_cells*i];
#endif // NO_SHARED_DOFS
    values_dofs[i] = src[srcidx];
  }

  if(constraint_mask)
    resolve_hanging_nodes_pmem<dim,fe_degree,NOTRANSPOSE>(values_dofs,constraint_mask);
}


//=============================================================================
// Distribute local to global
//=============================================================================

template <typename Number, int dim, int fe_degree>
__device__ void FEEvaluationGpu<Number,dim,fe_degree>::distribute_local_to_global(Number *dst)
{
  if(constraint_mask)
    resolve_hanging_nodes_pmem<dim,fe_degree,TRANSPOSE>(values_dofs,constraint_mask);

#ifdef NO_SHARED_DOFS
  for(int i=0; i<n_local_dofs; ++i) {
    const unsigned int dstidx = i*n_cells+cellid;
    dst[dstidx] += values_dofs[i];
  }
#else // NO_SHARED_DOFS
  if(use_coloring) {

    for(int i=0; i<n_local_dofs; ++i) {
      const unsigned int dstidx = loc2glob[i*n_cells];
      dst[dstidx] += values_dofs[i];
    }
  }
  else {
    for(int i=0; i<n_local_dofs; ++i) {
      const unsigned int dstidx = loc2glob[i*n_cells];

#ifdef MATRIX_FREE_NOTHING
      dst[dstidx] += values_dofs[i];
#else
      atomicAdd(&dst[dstidx],values_dofs[i]);
#endif
    }
  }
#endif // NO_SHARED_DOFS
}


#endif
