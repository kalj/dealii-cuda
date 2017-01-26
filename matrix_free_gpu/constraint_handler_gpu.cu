

#include "constraint_handler_gpu.h"
#include "cuda_utils.cuh"
#include "gpu_vec.h"

#define MATRIX_FREE_BKSIZE_CONSTR 128

DEAL_II_NAMESPACE_OPEN




namespace kernels
{

  template <typename Number>
  __global__ void set_constrained_dofs_kernel (Number              *dst,
                                               Number              val,
                                               const unsigned int  *constrained_dofs,
                                               const unsigned int   n_constrained_dofs);


  template <typename Number>
  __global__ void save_constrained_dofs_kernel (Number              *in,
                                                Number              *tmp_in,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs);

  template <typename Number>
  __global__ void save_constrained_dofs_kernel (const Number        *out,
                                                Number              *in,
                                                Number              *tmp_out,
                                                Number              *tmp_in,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs);

  template <typename Number>
  __global__ void load_constrained_dofs_kernel (Number              *out,
                                                const Number        *tmp_out,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs);

  template <typename Number>
  __global__ void load_and_add_constrained_dofs_kernel (Number              *out,
                                                        Number              *in,
                                                        const Number        *tmp_out,
                                                        const Number        *tmp_in,
                                                        const unsigned int  *constrained_dofs,
                                                        const unsigned int   n_constrained_dofs);
}



template <typename Number>
void ConstraintHandlerGpu<Number>::reinit_kernel_parameters()
{

  const unsigned int n_constrained_dofs = constrained_indices.size();
  const unsigned int constr_num_blocks = ceil(n_constrained_dofs / float(MATRIX_FREE_BKSIZE_CONSTR));
  const unsigned int constr_x_num_blocks = round(sqrt(constr_num_blocks)); // get closest to even square.
  const unsigned int constr_y_num_blocks = ceil(double(constr_num_blocks)/constr_x_num_blocks);

  constr_grid_dim = dim3(constr_x_num_blocks,constr_y_num_blocks);
  constr_block_dim = dim3(MATRIX_FREE_BKSIZE_CONSTR);
}

template <typename Number>
void ConstraintHandlerGpu<Number>
::reinit(const ConstraintMatrix &constraints,
         const unsigned int n_dofs)
{
  n_constrained_dofs = constraints.n_constraints();

  std::vector<unsigned int> constrained_dofs_host(n_constrained_dofs);

  unsigned int iconstr = 0;
  for(unsigned int i=0; i<n_dofs; i++) {
    if(constraints.is_constrained(i)) {
      constrained_dofs_host[iconstr] = i;
      iconstr++;
    }
  }

  constrained_indices = constrained_dofs_host;

  constrained_values_dst.reinit(n_constrained_dofs);
  constrained_values_src.reinit(n_constrained_dofs);

  // no edge constraints -- these are now hanging node constraints and are
  // handled by MatrixFreeGpu
  edge_indices.clear();

  reinit_kernel_parameters();
}


template <typename Number>
void ConstraintHandlerGpu<Number>
::reinit(const MGConstrainedDoFs  &mg_constrained_dofs,
         const unsigned int        level)
{
  std::vector<types::global_dof_index> indices;
  IndexSet index_set;

  // first set up list of DoFs on refinement edges
  index_set = mg_constrained_dofs.get_refinement_edge_indices(level);
  index_set.fill_index_vector(indices);
  edge_indices = indices;

  // then add also boundary DoFs to get all constrained DoFs
  index_set.add_indices(mg_constrained_dofs.get_boundary_indices(level));
  index_set.fill_index_vector(indices);
  constrained_indices = indices;


  n_constrained_dofs = constrained_indices.size();

  constrained_values_dst.reinit(n_constrained_dofs);
  constrained_values_src.reinit(n_constrained_dofs);

  reinit_kernel_parameters();
}


template <typename Number>
void ConstraintHandlerGpu<Number>::set_constrained_values(GpuVector <Number> &dst,
                                                          Number val) const
{
  if(n_constrained_dofs != 0) {
    kernels::set_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(dst.getData(),
                                                                                        val,
                                                                                        constrained_indices.getDataRO(),
                                                                                        n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::save_constrained_values(GpuVector<Number>        &src)
{
  if(n_constrained_dofs != 0) {
    kernels::save_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(src.getData(),
                                                                                         constrained_values_src.getData(),
                                                                                         constrained_indices.getDataRO(),
                                                                                         n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}


template <typename Number>
void ConstraintHandlerGpu<Number>::save_constrained_values(const GpuVector <Number> &dst,
                                                           GpuVector<Number>        &src)
{
  if(n_constrained_dofs != 0) {
    kernels::save_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(dst.getDataRO(),
                                                                                         src.getData(),
                                                                                         constrained_values_dst.getData(),
                                                                                         constrained_values_src.getData(),
                                                                                         constrained_indices.getDataRO(),
                                                                                         n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::load_constrained_values(GpuVector <Number>          &src) const
{
  if(n_constrained_dofs != 0) {
    kernels::load_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(src.getData(),
                                                                                         constrained_values_src.getDataRO(),
                                                                                         constrained_indices.getDataRO(),
                                                                                         n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}


template <typename Number>
void ConstraintHandlerGpu<Number>::load_and_add_constrained_values(GpuVector <Number>          &dst,
                                                                   GpuVector<Number>           &src) const
{
  if(n_constrained_dofs != 0) {
    kernels::load_and_add_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(dst.getData(),
                                                                                                 src.getData(),
                                                                                                 constrained_values_dst.getDataRO(),
                                                                                                 constrained_values_src.getDataRO(),
                                                                                                 constrained_indices.getDataRO(),
                                                                                                 n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::copy_edge_values(GpuVector <Number>          &dst,
                                                    const GpuVector<Number>           &src) const
{
  copy_with_indices(dst,src,edge_indices,edge_indices);
}

namespace kernels
{

  template <typename Number>
  __global__ void set_constrained_dofs_kernel (Number               *dst,
                                               Number               val,
                                               const unsigned int   *constrained_dofs,
                                               const unsigned int   n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
    if(dof < n_constrained_dofs) {
      dst[constrained_dofs[dof]] = val;
    }
  }

  template <typename Number>
  __global__ void save_constrained_dofs_kernel (Number              *in,
                                                Number              *tmp_in,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
    if(dof < n_constrained_dofs) {
      tmp_in[dof]  = in[constrained_dofs[dof]];
      in[constrained_dofs[dof]] = 0;
    }
  }


  template <typename Number>
  __global__ void save_constrained_dofs_kernel (const Number        *out,
                                                Number              *in,
                                                Number              *tmp_out,
                                                Number              *tmp_in,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
    if(dof < n_constrained_dofs) {
      tmp_out[dof] = out[constrained_dofs[dof]];
      tmp_in[dof]  = in[constrained_dofs[dof]];
      in[constrained_dofs[dof]] = 0;
    }
  }

  template <typename Number>
  __global__ void load_constrained_dofs_kernel (Number              *in,
                                                const Number        *tmp_in,
                                                const unsigned int  *constrained_dofs,
                                                const unsigned int   n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
    if(dof < n_constrained_dofs) {
      in[constrained_dofs[dof]]  = tmp_in[dof];
    }
  }


  template <typename Number>
  __global__ void load_and_add_constrained_dofs_kernel (Number              *out,
                                                        Number              *in,
                                                        const Number        *tmp_out,
                                                        const Number        *tmp_in,
                                                        const unsigned int  *constrained_dofs,
                                                        const unsigned int   n_constrained_dofs)
  {
    const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
    if(dof < n_constrained_dofs) {
      out[constrained_dofs[dof]] = tmp_out[dof] + tmp_in[dof];
      in[constrained_dofs[dof]]  = tmp_in[dof];
    }
  }
}


//=============================================================================
// explicit instantiation
//=============================================================================

template class ConstraintHandlerGpu<float>;
template class ConstraintHandlerGpu<double>;

DEAL_II_NAMESPACE_CLOSE