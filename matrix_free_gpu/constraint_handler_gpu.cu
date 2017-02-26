

#include "constraint_handler_gpu.h"
#include "cuda_utils.cuh"


DEAL_II_NAMESPACE_OPEN

#define CONSTRAINT_OPS_BKSIZE 256


namespace kernels
{

  template <typename Number>
  __global__ void set_constrained_dofs (Number              *dst,
                                        Number              val,
                                        const unsigned int  *constrained_dofs,
                                        const unsigned int   n_constrained_dofs);


  template <typename Number>
  __global__ void save_constrained_dofs (Number              *in,
                                         Number              *tmp_in,
                                         const unsigned int  *constrained_dofs,
                                         const unsigned int   n_constrained_dofs);

  template <typename Number>
  __global__ void save_constrained_dofs (const Number        *out,
                                         Number              *in,
                                         Number              *tmp_out,
                                         Number              *tmp_in,
                                         const unsigned int  *constrained_dofs,
                                         const unsigned int   n_constrained_dofs);

  template <typename Number>
  __global__ void load_and_add_constrained_dofs (Number              *out,
                                                 Number              *in,
                                                 const Number        *tmp_out,
                                                 const Number        *tmp_in,
                                                 const unsigned int  *constrained_dofs,
                                                 const unsigned int   n_constrained_dofs);
}



template <typename Number>
void ConstraintHandlerGpu<Number>::reinit_kernel_parameters()
{
  for(int part=0; part<n_partitions; ++part) {
    const unsigned int n_constrained_dofs = constrained_indices.local_size(part);
    const unsigned int num_blocks = ceil(n_constrained_dofs / float(CONSTRAINT_OPS_BKSIZE));
    const unsigned int x_num_blocks = round(sqrt(num_blocks)); // get closest to even square.
    const unsigned int y_num_blocks = ceil(double(num_blocks)/x_num_blocks);

    grid_dim[part] = dim3(x_num_blocks,y_num_blocks);
  }
  block_dim = dim3(CONSTRAINT_OPS_BKSIZE);
}

template <typename Number>
void ConstraintHandlerGpu<Number>
::reinit(const std::vector<ConstraintMatrix> &constraints,
         const std::shared_ptr<const GpuPartitioner> &partitioner)
{
  n_partitions = partitioner->n_partitions();
  n_constrained_dofs.resize(n_partitions);

  std::vector<std::vector<unsigned int> > constrained_dofs_host(n_partitions);

  for(int part=0; part<n_partitions; ++part) {

    n_constrained_dofs[part] = constraints[part].n_constraints();

    constrained_dofs_host[part].resize(n_constrained_dofs[part]);

    unsigned int iconstr = 0;
    for(unsigned int i=0; i<partitioner->n_dofs(part); i++) {
      if(constraints[part].is_constrained(i)) {
        constrained_dofs_host[part][iconstr] = i;
        iconstr++;
      }
    }
  }

  constrained_indices.reinit(partitioner,n_constrained_dofs);
  constrained_indices = constrained_dofs_host;

  constrained_values_dst.reinit(partitioner,n_constrained_dofs);
  constrained_values_src.reinit(partitioner,n_constrained_dofs);

  // no edge constraints -- these are now hanging node constraints and are
  // handled by MatrixFreeGpu
  edge_indices.clear();

  reinit_kernel_parameters();
}


template <typename Number>
void ConstraintHandlerGpu<Number>
::reinit(const std::vector<MGConstrainedDoFs> &mg_constrained_dofs,
         const std::shared_ptr<const GpuPartitioner> &partitioner,
         const unsigned int        level)
{
  n_partitions = partitioner->n_partitions();
  n_constrained_dofs.resize(n_partitions);

  std::vector<std::vector<unsigned int> > constrained_indices_host(n_partitions);
  std::vector<std::vector<unsigned int> > edge_indices_host(n_partitions);
  std::vector<unsigned int>               n_edge_dofs(n_partitions);

  for(int part=0; part<n_partitions; ++part) {

    IndexSet index_set;

  // first set up list of DoFs on refinement edges
    index_set = mg_constrained_dofs[part].get_refinement_edge_indices(level);
    index_set.fill_index_vector(edge_indices_host[part]);
    n_edge_dofs[part] = edge_indices_host[part].size();
    // edge_indices = indices;

    // then add also boundary DoFs to get all constrained DoFs
    index_set.add_indices(mg_constrained_dofs[part].get_boundary_indices(level));
    index_set.fill_index_vector(constrained_indices_host[part]);

    n_constrained_dofs[part] = constrained_indices_host[part].size();
  }

  constrained_indices.reinit(partitioner,n_constrained_dofs);
  constrained_indices = constrained_indices_host;

  edge_indices.reinit(partitioner,n_edge_dofs);
  edge_indices = edge_indices_host;

  constrained_values_dst.reinit(partitioner,n_constrained_dofs);
  constrained_values_src.reinit(partitioner,n_constrained_dofs);

  reinit_kernel_parameters();
}


template <typename Number>
void ConstraintHandlerGpu<Number>::set_constrained_values(MultiGpuVector <Number> &dst,
                                                          Number val) const
{
  for(int i=0; i<n_partitions; ++i) {
    if(n_constrained_dofs[i] != 0) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      kernels::set_constrained_dofs<Number> <<<grid_dim[i],block_dim>>>(dst.getData(i),
                                                                        val,
                                                                        constrained_indices.getDataRO(i),
                                                                        n_constrained_dofs[i]);
      CUDA_CHECK_LAST;
    }
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::save_constrained_values(MultiGpuVector<Number>        &src)
{
  for(int i=0; i<n_partitions; ++i) {
    if(n_constrained_dofs[i] != 0) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      kernels::save_constrained_dofs<Number> <<<grid_dim[i],block_dim>>>(src.getData(i),
                                                                         constrained_values_src.getData(i),
                                                                         constrained_indices.getDataRO(i),
                                                                         n_constrained_dofs[i]);
      CUDA_CHECK_LAST;
    }
  }
}


template <typename Number>
void ConstraintHandlerGpu<Number>::save_constrained_values(const MultiGpuVector <Number> &dst,
                                                           MultiGpuVector<Number>        &src)
{
  for(int i=0; i<n_partitions; ++i) {
    if(n_constrained_dofs[i] != 0) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      kernels::save_constrained_dofs<Number> <<<grid_dim[i],block_dim>>>(dst.getDataRO(i),
                                                                         src.getData(i),
                                                                         constrained_values_dst.getData(i),
                                                                         constrained_values_src.getData(i),
                                                                         constrained_indices.getDataRO(i),
                                                                         n_constrained_dofs[i]);
      CUDA_CHECK_LAST;
    }
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::load_constrained_values(MultiGpuVector <Number>          &src) const
{
  copy_with_indices(src,constrained_indices,constrained_values_src);
}


template <typename Number>
void ConstraintHandlerGpu<Number>::load_and_add_constrained_values(MultiGpuVector <Number>          &dst,
                                                                   MultiGpuVector<Number>           &src) const
{
  for(int i=0; i<n_partitions; ++i) {
    if(n_constrained_dofs[i] != 0) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      kernels::load_and_add_constrained_dofs<Number> <<<grid_dim[i],block_dim>>>(dst.getData(i),
                                                                                 src.getData(i),
                                                                                 constrained_values_dst.getDataRO(i),
                                                                                 constrained_values_src.getDataRO(i),
                                                                                 constrained_indices.getDataRO(i),
                                                                                 n_constrained_dofs[i]);
      CUDA_CHECK_LAST;
    }
  }
}

template <typename Number>
void ConstraintHandlerGpu<Number>::copy_edge_values(MultiGpuVector <Number>          &dst,
                                                    const MultiGpuVector<Number>     &src) const
{
  if(edge_indices.size() > 0) {
    copy_with_indices(dst,edge_indices,src,edge_indices);
  }
}


template <typename Number>
std::size_t ConstraintHandlerGpu<Number>::memory_consumption () const
{
  std::size_t memory = 0;
  memory += MemoryConsumption::memory_consumption(n_constrained_dofs);
  memory += MemoryConsumption::memory_consumption(constrained_indices);
  memory += MemoryConsumption::memory_consumption(edge_indices);
  memory += MemoryConsumption::memory_consumption(constrained_values_src);
  memory += MemoryConsumption::memory_consumption(constrained_values_dst);
  memory += sizeof(grid_dim);
  memory += sizeof(block_dim);
  return memory;
}


namespace kernels
{

  template <typename Number>
  __global__ void set_constrained_dofs (Number               *dst,
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
  __global__ void save_constrained_dofs (Number              *in,
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
  __global__ void save_constrained_dofs (const Number        *out,
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
  __global__ void load_and_add_constrained_dofs (Number              *out,
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