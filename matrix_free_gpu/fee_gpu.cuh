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

#ifdef MATRIX_FREE_HANGING_NODES
#include "hanging_nodes.cuh"
#endif
//=============================================================================
// Object which lives on the Gpu which contains methods for element local
// operations, such as gradient and value evaluation. This is created for each
// element at each multiplication, but since creation only means setting up a
// few pointers, overhead is small.
//=============================================================================


// This is a base class containing common code and data. Subclasses implement
// most operations.
template <int dim, int fe_degree, typename Number>
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

  // information on hanging node constraints
#ifdef MATRIX_FREE_HANGING_NODES
  const unsigned int constraint_mask;
#endif

public:

  const GpuArray<dim,Number> *quadrature_points;

  __device__ FEEvaluationGpuBase(int cellid, const data_type *data)
    :
    n_cells(data->n_cells),
    cellid(cellid),
#ifdef MATRIX_FREE_HANGING_NODES
    constraint_mask(data->constraint_mask[cellid]),
#endif
    use_coloring(data->use_coloring)
  {
  }

};


//=============================================================================
// shmem version
//=============================================================================

// This implementation uses one thread per DoF in an element, and implements
// this using Shared Memory. (PIE == Parallel In Element), or actually, no
// suffix

template <int dim, int fe_degree, typename Number>
class FEEvaluationGpu : public FEEvaluationGpuBase<dim,fe_degree,Number>
{
public:
  typedef Number number_type;
  typedef typename MatrixFreeGpu<dim,Number>::GpuData data_type;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::n_dofs_1d;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::n_q_points_1d;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::n_local_dofs;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::n_q_points;
  typedef GpuArray<dim,Number> gradient_type;


private:
  using FEEvaluationGpuBase<dim,fe_degree,Number>::loc2glob;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::inv_jac;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::JxW;

  using FEEvaluationGpuBase<dim,fe_degree,Number>::cellid;
  using FEEvaluationGpuBase<dim,fe_degree,Number>::n_cells;

  using FEEvaluationGpuBase<dim,fe_degree,Number>::use_coloring;
#ifdef MATRIX_FREE_HANGING_NODES
  using FEEvaluationGpuBase<dim,fe_degree,Number>::constraint_mask;
#endif


  // internal buffers
  Number             *values;
  Number             *gradients[dim];

  const unsigned int rowlength;
  const unsigned int rowstart;
public:
  const GpuArray<dim,Number> *quadrature_points;

  // Constructor. Requires the cell id, the pointer to the data cache, and the
  // pointer to the element local shared scratch data.
  __device__ FEEvaluationGpu(int cellid, const data_type *data, SharedData<dim,Number> *shdata);

  // read local values from the device-side array
  __device__ void read_dof_values(const Number *src);
  // write them back
  __device__ void distribute_local_to_global(Number *dst);

  // evaluate function values and gradients on quadrature points
  __device__ void evaluate(const bool evaluate_val,
                           const bool evaluate_grad);


  // specify value or gradient at dof
  __device__ void submit_dof_value(const Number &val, const unsigned int i);
  // __device__ void submit_gradient(const gradient_type &grad, const unsigned int q);

  // specify value or gradient at quadrature point
  __device__ void submit_value(const Number &val, const unsigned int q);
  __device__ void submit_gradient(const gradient_type &grad, const unsigned int q);

  // get value or gradient at quadrature point
  __device__ Number get_value(const unsigned int q) const ;
  __device__ gradient_type get_gradient(const unsigned int q) const ;

  // get position of quadrature point
  __device__  const GpuArray<dim,Number> &get_quadrature_point(const unsigned int q) const {
    return quadrature_points[q*n_cells];
  }

  // integrate function values and/or gradients
  __device__ void integrate(const bool integrate_val, const bool integrate_grad);

  // return the global index of local quadrature point q
  __device__ unsigned int get_global_q(const unsigned int q) const
  {
    return rowstart+q;
  }

  // apply the function lop->quad_operation on all quadrature points (i.e. hide
  // particularities of how to loop over quadrature points from user).
  template <typename LocOp>
  __device__ void apply_quad_point_operations(const LocOp *lop);

};

template <int dim, int fe_degree, typename Number>
__device__ FEEvaluationGpu<dim,fe_degree,Number>::FEEvaluationGpu(int cellid,
                                                                  const data_type *data,
                                                                  SharedData<dim,Number> *shdata)
  :
  FEEvaluationGpuBase<dim,fe_degree,Number>(cellid,data),
  rowlength(data->rowlength),
  rowstart(data->rowstart + cellid*rowlength)
{
  values = shdata->values;
  loc2glob = data->loc2glob+rowlength*cellid;
#ifdef MATRIX_FREE_UNIFORM_MESH
  inv_jac = data->inv_jac+cellid;
#else
  inv_jac = data->inv_jac+rowlength*cellid;
#endif
  JxW = data->JxW+rowlength*cellid;
  quadrature_points = data->quadrature_points+rowlength*cellid;

  for(int d = 0; d < dim; ++d) {
    gradients[d] = shdata->gradients[d];
  }
}


template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::evaluate(const bool evaluate_val,
                                                                const bool evaluate_grad)
{
  if(evaluate_grad)
    TensorOpsShmem<dim,fe_degree+1,Number>::grad_at_quad_pts (gradients,values);

  if(evaluate_val) {

    if(evaluate_grad) __syncthreads();
    TensorOpsShmem<dim,fe_degree+1,Number>::fun_at_quad_pts (values);
  }
  __syncthreads();
}

template <int dim, int fe_degree, typename Number>
__device__ Number
FEEvaluationGpu<dim,fe_degree,Number>::get_value(const unsigned int q) const
{
  return values[q];
}

template <int dim, int fe_degree, typename Number>
__device__ FEEvaluationGpu<dim,fe_degree,Number>::gradient_type
FEEvaluationGpu<dim,fe_degree,Number>::get_gradient(const unsigned int q) const
{
  // compute J^{-1} * gradients_quad[q]
  gradient_type grad;
#ifdef MATRIX_FREE_UNIFORM_MESH
  const Number* J = &inv_jac[0];
#else
  const Number* J = &inv_jac[q];
#endif


  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_UNIFORM_MESH
    grad[d1] = J[0]*gradients[d1][q];
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[rowlength*n_cells*(dim*d2+d1)] * gradients[d2][q];
    }
    grad[d1] = tmp;
#endif

  }

  return grad;
}

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::submit_dof_value(const Number &val, const unsigned int i)
{
  values[i] = val;
}

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::submit_value(const Number &val, const unsigned int q)
{
  const Number jxw = JxW[q];
  values[q] = jxw*val;
}

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::submit_gradient(const gradient_type &grad, const unsigned int q)
{
  // compute J^{-T} * grad * det(J) *w_q
#ifdef MATRIX_FREE_UNIFORM_MESH
  const Number* J = &inv_jac[0];
#else
  const Number* J = &inv_jac[q];
#endif
  const Number jxw = JxW[q];

  for(int d1=0; d1<dim; d1++) {
#ifdef MATRIX_FREE_UNIFORM_MESH
    gradients[d1][q] = grad[d1] * J[0] * jxw;
#else
    Number tmp = 0;
    for(int d2=0; d2<dim; d2++) {
      tmp += J[n_cells*rowlength*(dim*d1+d2)]*grad[d2];
    }
    gradients[d1][q] = tmp * jxw;
#endif

  }
}

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::integrate(const bool integrate_val,
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
  __syncthreads();
}


template <int dim, int fe_degree, typename Number>
template <typename LocOp>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::apply_quad_point_operations(const LocOp *lop)
{
  const unsigned int q = (dim==1 ? threadIdx.x%n_q_points_1d :
                          dim==2 ? threadIdx.x%n_q_points_1d + n_q_points_1d*threadIdx.y :
                          threadIdx.x%n_q_points_1d + n_q_points_1d*(threadIdx.y + n_q_points_1d*threadIdx.z));

  lop->quad_operation(this,q);
  __syncthreads();
}


//=============================================================================
// Read DoF values
//=============================================================================

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::read_dof_values(const Number *src)
{
  const unsigned int  idx = (threadIdx.x%n_dofs_1d)
    +(dim>1 ? threadIdx.y : 0)*n_dofs_1d
    +(dim>2 ? threadIdx.z : 0)*n_dofs_1d*n_dofs_1d;

  const unsigned int srcidx = loc2glob[idx];
  values[idx] = __ldg(&src[srcidx]);

#ifdef MATRIX_FREE_HANGING_NODES
    resolve_hanging_nodes_shmem<dim,fe_degree,NOTRANSPOSE>(values,constraint_mask);
#endif

  __syncthreads();
}



//=============================================================================
// Distribute local to global
//=============================================================================

template <int dim, int fe_degree, typename Number>
__device__ void FEEvaluationGpu<dim,fe_degree,Number>::distribute_local_to_global(Number *dst)
{
#ifdef MATRIX_FREE_HANGING_NODES
    resolve_hanging_nodes_shmem<dim,fe_degree,TRANSPOSE>(values,constraint_mask);
#endif

  {
    const unsigned int  i = (threadIdx.x%n_q_points_1d)
      +(dim>1 ? threadIdx.y : 0)*n_q_points_1d
      +(dim>2 ? threadIdx.z : 0)*n_q_points_1d*n_q_points_1d;
    const unsigned int dstidx = loc2glob[i];

    if(use_coloring)
      dst[dstidx] += values[i];
    else
      atomicAddWrapper(&dst[dstidx],values[i]);
  }

}

#endif
