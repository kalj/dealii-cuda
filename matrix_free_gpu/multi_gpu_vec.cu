/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)multi_gpu_vec.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#include <cfloat>

#include "multi_gpu_vec.h"
#include "atomic.cuh"
#include "cuda_utils.cuh"


#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8
#define VR_BKSIZE 512
#define VR_CHUNK_SIZE 8
#define COPY_WITH_INDEX_BKSIZE 256


using namespace dealii;

// #include <deal.II/lac/vector_memory.templates.h>
// template class VectorMemory<MultiGpuVector<double> >;
// template class GrowingVectorMemory<MultiGpuVector<double> >;
// template class VectorMemory<MultiGpuVector<float> >;
// template class GrowingVectorMemory<MultiGpuVector<float> >;

template <typename DstNumber, typename SrcNumber, typename ScalarNumber>
__global__ void vec_equ(DstNumber *v1, const SrcNumber *v2, const ScalarNumber a, const int N);


//=============================================================================
// Constructors / assignment
//=============================================================================

template <typename Number>
MultiGpuVector<Number>::DevRef& MultiGpuVector<Number>::DevRef::operator=(const Number value)
{
  CUDA_CHECK_SUCCESS(cudaSetDevice(owning_device));
  CUDA_CHECK_SUCCESS(cudaMemcpy(ptr,&value,sizeof(Number),
                                cudaMemcpyHostToDevice));
  return *this;
}

template <typename Number>
MultiGpuVector<Number>::MultiGpuVector(const std::shared_ptr<const GpuPartitioner> &partitioner_in)
  : vec(partitioner_in->n_partitions()),
    import_data(partitioner_in->n_partitions()),
    import_indices(partitioner_in->n_partitions()),
    local_size(partitioner_in->n_partitions()),
    global_size(partitioner_in->n_global_dofs()),
    partitioner(partitioner_in),
    vector_is_ghosted(false),
    vector_is_compressed(true)
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    local_size[i] = partitioner->n_dofs(i);
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_data[i],partitioner->n_import_indices(i)*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_indices[i],partitioner->n_import_indices(i)*sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(import_indices[i],partitioner->import_indices(i).data(),
                                  partitioner->n_import_indices(i)*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
  }
}

template <typename Number>
MultiGpuVector<Number>::MultiGpuVector(const MultiGpuVector<Number>& old)
  : vec(old.partitioner->n_partitions()),
    import_data(old.partitioner->n_partitions()),
    import_indices(old.partitioner->n_partitions()),
    local_size(old.partitioner->n_partitions()),
    global_size(old.partitioner->n_global_dofs()),
    partitioner(old.partitioner),
    vector_is_ghosted(old.vector_is_ghosted),
    vector_is_compressed(old.vector_is_compressed)
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    local_size[i] = partitioner->n_dofs(i);
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_data[i],partitioner->n_import_indices(i)*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_indices[i],partitioner->n_import_indices(i)*sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(import_indices[i],partitioner->import_indices(i).data(),
                                  partitioner->n_import_indices(i)*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

    CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],old.vec[i],ghosted_size*sizeof(Number),
                                  cudaMemcpyDeviceToDevice));
  }
}

template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const Vector<Number>& old_cpu)
{
  AssertDimension(global_size, old_cpu.size());

  const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                  cpu_data+partitioner->local_dof_offset(i),
                                  local_size[i]*sizeof(Number),
                                  cudaMemcpyHostToDevice));
  }

  vector_is_ghosted = false;
  vector_is_compressed = true;

  return *this;
}

template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const std::vector<Number>& old_cpu)
{
  AssertDimension(global_size, old_cpu.size());

  const Number *cpu_data = old_cpu.data();

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                  cpu_data+partitioner->local_dof_offset(i),
                                  local_size[i]*sizeof(Number),
                                  cudaMemcpyHostToDevice));
  }

  vector_is_ghosted = false;
  vector_is_compressed = true;

  return *this;
}

template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const MultiGpuVector<Number>& old)
{
  AssertDimension(global_size, old.size());
  // FIXME: also check for compatible partitioners

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                  old.vec[i],
                                  ghosted_size*sizeof(Number),
                                  cudaMemcpyDeviceToDevice));
  }

  vector_is_ghosted = old.vector_is_ghosted;
  vector_is_ghosted = old.vector_is_compressed;

  return *this;
}

// copy constructor from vector based on other number type
template <typename Number>
template <typename OtherNumber>
MultiGpuVector<Number>::MultiGpuVector(const MultiGpuVector<OtherNumber>& old)
  : vec(old.partitioner->n_partitions()),
    import_data(old.partitioner->n_partitions()),
    import_indices(old.partitioner->n_partitions()),
    local_size(old.partitioner->n_partitions()),
    global_size(old.partitioner->n_global_dofs()),
    partitioner(old.partitioner),
    vector_is_ghosted(old.vector_is_ghosted),
    vector_is_compressed(old.vector_is_compressed)
{

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    local_size[i] = partitioner->n_dofs(i);
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);

    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_data[i],partitioner->n_import_indices(i)*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_indices[i],partitioner->n_import_indices(i)*sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(import_indices[i],partitioner->import_indices(i).data(),
                                  partitioner->n_import_indices(i)*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));


    const int nblocks = 1 + (ghosted_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_equ<Number,OtherNumber,double> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i], old.vec[i],
                                                                        1.0,ghosted_size);
    CUDA_CHECK_LAST;

  }
}

// same for assignment
template <typename Number>
template <typename OtherNumber>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const MultiGpuVector<OtherNumber>& old)
{
  AssertDimension(global_size, old.size());
  // FIXME: also check for compatible partitioners

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (ghosted_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_equ<Number,OtherNumber,double> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i], old.vec[i],
                                                                        1.0,ghosted_size);
    CUDA_CHECK_LAST;
  }

  vector_is_ghosted = old.vector_is_ghosted;
  vector_is_compressed = old.vector_is_compressed;

  return *this;
}


template <typename Number>
MultiGpuVector<Number>::~MultiGpuVector()
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    if(vec[i] != NULL) {
      CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
      CUDA_CHECK_SUCCESS(cudaFree(import_data[i]));
    }
  }
}

template <typename Number>
void MultiGpuVector<Number>::copyToHost(Vector<Number>& dst) const
{
  AssertDimension(global_size, dst.size());

  // FIXME: probably check that this vector is compressed

  Number *cpu_data = &dst[0];

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(cpu_data+partitioner->local_dof_offset(i),
                                  vec[i],
                                  local_size[i]*sizeof(Number),
                                  cudaMemcpyDeviceToHost));
  }
}

// initialize with a partitioner
template <typename Number>
void MultiGpuVector<Number>::reinit (const std::shared_ptr<const GpuPartitioner> &partitioner_in)
{
  // if we already have same or equivalent partitioner, just return
  if(partitioner->is_compatible(*partitioner_in))
    return;

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    if(vec[i] != NULL) {
      CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
      CUDA_CHECK_SUCCESS(cudaFree(import_data[i]));
      CUDA_CHECK_SUCCESS(cudaFree(import_indices[i]));
    }
  }

  vec.resize(partitioner_in->n_partitions());
  import_data.resize(partitioner_in->n_partitions());
  import_indices.resize(partitioner_in->n_partitions());
  local_size.resize(partitioner_in->n_partitions());
  global_size = partitioner_in->n_global_dofs();
  partitioner = partitioner_in;
  vector_is_ghosted = false;
  vector_is_compressed = true;

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    local_size[i] = partitioner->n_dofs(i);
    const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);

    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_data[i],
                                  partitioner->n_import_indices(i)*sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMalloc(&import_indices[i],
                                  partitioner->n_import_indices(i)*sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(import_indices[i],partitioner->import_indices(i).data(),
                                  partitioner->n_import_indices(i)*sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

    CUDA_CHECK_SUCCESS(cudaMemset(vec[i], 0, local_size[i]*sizeof(Number)));
  }
}


// resize to have the same structure as the one provided and/or clear
// vector. note that the second argument must have a default value equal to
// false
template <typename Number>
void MultiGpuVector<Number>::reinit (const MultiGpuVector<Number>& other,
                                bool leave_elements_uninitialized)
{
  // if we already have same or equivalent partitioner, just return
  if(!partitioner->is_compatible(*other.partitioner)) {
    // clean up
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      if(vec[i] != NULL) {
        CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
        CUDA_CHECK_SUCCESS(cudaFree(import_data[i]));
        CUDA_CHECK_SUCCESS(cudaFree(import_indices[i]));
      }
    }

    // reinit partitioning
    vec.resize(other.partitioner->n_partitions());
    import_data.resize(other.partitioner->n_partitions());
    import_indices.resize(other.partitioner->n_partitions());
    local_size.resize(other.partitioner->n_partitions());

    global_size = other.partitioner->n_global_dofs();
    partitioner = other.partitioner;

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      local_size[i] = partitioner->n_dofs(i);
      const unsigned int ghosted_size = local_size[i] + partitioner->n_ghost_dofs_tot(i);
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&import_data[i],partitioner->n_import_indices(i)*sizeof(Number)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&import_indices[i],partitioner->n_import_indices(i)*sizeof(unsigned int)));

      CUDA_CHECK_SUCCESS(cudaMemcpy(import_indices[i],partitioner->import_indices(i).data(),
                                    partitioner->n_import_indices(i)*sizeof(unsigned int),
                                    cudaMemcpyHostToDevice));
    }
  }

  if(!leave_elements_uninitialized) {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(i));
      CUDA_CHECK_SUCCESS(cudaMemset(vec[i], 0, local_size[i]*sizeof(Number)));
    }
  }

  vector_is_ghosted = false;
  vector_is_compressed = true;
}


template <typename Number>
Number MultiGpuVector<Number>::operator()(const size_t i) const
{
  // FIXME: probably check for compressed

  const int owning_device = partitioner->dof_owner(i);
  const int local_index = partitioner->local_index(i);

  Number value;

  CUDA_CHECK_SUCCESS(cudaSetDevice(i));
  CUDA_CHECK_SUCCESS(cudaMemcpy(&value,vec[owning_device]+local_index,sizeof(Number),
                                cudaMemcpyDeviceToHost));
  return value;
}

// necessary for deal.ii but shouldn't be used!
template <typename Number>
MultiGpuVector<Number>::DevRef MultiGpuVector<Number>::operator()(const size_t i)
{
  // FIXME: probably check for compressed / invalidate ghosted state

  const int owning_device = partitioner->dof_owner(i);
  const int local_index = partitioner->local_index(i);
  return MultiGpuVector<Number>::DevRef(vec[owning_device]+local_index,owning_device);
}


//=============================================================================
// Element wise operations (mult. with scalar, vector addition)
//=============================================================================


template <typename Number>
__global__ void vec_sadd(Number *v1, const Number *v2, const Number a, const Number b, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v1[idx] = a*v1[idx] + b*v2[idx];
  }
}

struct Binop_Multiplication {
  template <typename Number>
  __device__ static inline Number operation(const Number a, const Number b) { return a*b; }
};

struct Binop_Division {
  template <typename Number>
  __device__ static inline Number operation(const Number a, const Number b) { return a/b; }
};

template <typename Number, typename Binop>
__global__ void vec_bin_op(Number *v1, const Number *v2, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v1[idx] = Binop::operation(v1[idx],v2[idx]);
  }
}


template <typename DstNumber, typename SrcNumber, typename ScalarNumber>
__global__ void vec_equ(DstNumber *v1, const SrcNumber *v2, const ScalarNumber a, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v1[idx] = a*v2[idx];
  }
}

template <typename Number>
__global__ void vec_scale(Number *v, const Number a, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v[idx] *= a;
  }
}


template <typename Number>
__global__ void vec_init(Number *v, const Number a, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v[idx] = a;
  }
}

template <typename Number>
__global__ void vec_invert(Number *v, const int N)
{
  const int idx_base = threadIdx.x + blockIdx.x*(blockDim.x*CHUNKSIZE_ELEMWISE_OP);

  for(int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c) {
    const int idx=idx_base+c*BKSIZE_ELEMWISE_OP;
    if(idx < N)
      v[idx] = 1.0/v[idx];
  }
}


// scaled addition of vectors
template <typename Number>
void MultiGpuVector<Number>::sadd (const Number a,
                              const Number b,
                              const MultiGpuVector<Number> &x)
{

  AssertDimension(global_size, x.size());
  // FIXME: also check for compatible partitioners
  // FIXME: probably check for compressed state, and invalidate ghosted

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_sadd<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],a,b,local_size[i]);
    CUDA_CHECK_LAST;
  }
}



// element-wise multiplication of vectors
template <typename Number>
void MultiGpuVector<Number>::scale (const MultiGpuVector<Number> &x)
{
  AssertDimension(global_size, x.size());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_bin_op<Number,Binop_Multiplication> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],
                                                                             local_size[i]);
    CUDA_CHECK_LAST;
  }
}

// element-wise division of vectors
template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator/= (const MultiGpuVector<Number> &x)
{
  AssertDimension(global_size, x.size());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_bin_op<Number,Binop_Division> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],
                                                                       local_size[i]);
    CUDA_CHECK_LAST;
  }
  return *this;
}

template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::invert()
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_invert<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],local_size[i]);
    CUDA_CHECK_LAST;
  }
  return *this;
}

// scaled assignment of a vector
template <typename Number>
void MultiGpuVector<Number>::equ (const Number a,
                             const MultiGpuVector<Number> &x)
{
  AssertDimension(global_size, x.size());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_equ<Number,Number,Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],a,
                                                                   local_size[i]);
    CUDA_CHECK_LAST;
  }
}


// scale the elements of the vector
// by a fixed value
template <typename Number>
MultiGpuVector<Number> & MultiGpuVector<Number>::operator *= (const Number a)
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_scale<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],a,local_size[i]);
    CUDA_CHECK_LAST;
  }
  return *this;
}


// return the l2 norm of the vector
template <typename Number>
Number MultiGpuVector<Number>::l2_norm () const {
  return sqrt((*this)*(*this));
}


// initialize vector with value
template <typename Number>
MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const Number a)
{
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    const int nblocks = 1 + (local_size[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
    vec_init<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],a,local_size[i]);
    CUDA_CHECK_LAST;
  }

  return *this;
}

//=============================================================================
// Reduction operations
//=============================================================================

/*
 These are implemented using a combination of some general utilities plus a
 struct containing the specific operations of the reduction you want to
 implement. For instance, if we want to implement a max operation, the following
 struct should be used:

 template <typename Number>
 struct MaxOperation {
   typedef Number shmem_type;
   typedef Number result_type;
   __device__ static Number elemwise_op(const Number v) {return v == 0;}
   __device__ static Number red_op(const Number a, const Number b){return a>b?a:b;}
   __device__ static Number atomic_op(Number *dst, const Number a){return atomicMax(dst,a);}
   __device__ static Number ident_value() { return -DBL_MAX; }
 };

 Then, this is fed to the single_vec_reduction kernel. It works similarly for
 dual vector reductions.
*/


// reduction helper functions

template <typename Operation, typename Number>
__device__ void reduceWithinWarp(volatile typename Operation::shmem_type *res_buf, int local_idx) {
  if(VR_BKSIZE >= 64) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+32]);
  if(VR_BKSIZE >= 32) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+16]);
  if(VR_BKSIZE >= 16) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+8]);
  if(VR_BKSIZE >= 8)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+4]);
  if(VR_BKSIZE >= 4)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+2]);
  if(VR_BKSIZE >= 2)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+1]);
}

template <typename Operation, typename Number>
__device__ void reduce(typename Operation::result_type *res, typename Operation::shmem_type *res_buf,
                      const unsigned int local_idx, const unsigned int global_idx, const unsigned int N)
{
  for(int s = VR_BKSIZE/2; s>32; s=s>>1) {

    if(local_idx < s)
      res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+s]);
    __syncthreads();
  }

  if(local_idx < 32) {
    reduceWithinWarp<Operation,Number>(res_buf,local_idx);
  }

  if(local_idx == 0)
    Operation::atomic_op(res,res_buf[0]);
}

// single vector reductions, e.g. all_zero, max, min, sum, prod, etc

template <typename Operation, typename Number>
__global__ void single_vec_reduction(typename Operation::result_type *res, const Number *v, const int N)
{
  __shared__ typename Operation::shmem_type res_buf[VR_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*VR_CHUNK_SIZE);
  const int local_idx = threadIdx.x;


  if(global_idx < N)
    res_buf[local_idx] = Operation::elemwise_op(v[global_idx]);
  else
    res_buf[local_idx] = Operation::ident_value();

  for(int c = 1; c < VR_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*VR_BKSIZE;
    if(idx < N)
      res_buf[local_idx] = Operation::red_op(res_buf[local_idx],
                                             Operation::elemwise_op(v[idx]));
  }

  __syncthreads();

  reduce<Operation,Number> (res,res_buf,local_idx,global_idx,N);
}

// single vector reductions, e.g. dot product

template <typename Operation, typename Number>
__global__ void dual_vec_reduction(typename Operation::result_type *res, const Number *v1,
                                   const Number *v2, const int N)
{
  __shared__ typename Operation::shmem_type res_buf[VR_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*VR_CHUNK_SIZE);
  const int local_idx = threadIdx.x;

  if(global_idx < N)
    res_buf[local_idx] = Operation::binary_op(v1[global_idx],v2[global_idx]);
  else
    res_buf[local_idx] = Operation::ident_value();

  for(int c = 1; c < VR_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*VR_BKSIZE;
    if(idx < N)
      res_buf[local_idx] = Operation::red_op(res_buf[local_idx],
                                             Operation::binary_op(v1[idx],v2[idx]));
  }

  __syncthreads();

  reduce<Operation,Number> (res,res_buf,local_idx,global_idx,N);
}

// all_zero

template <typename Number>
struct AllZero {
  typedef bool shmem_type;
  typedef unsigned int result_type;
  __device__ static bool elemwise_op(const Number v) {return v == 0;}
  __device__ static bool red_op(const bool a, const bool b){return a && b;}
  __device__ static unsigned int atomic_op(unsigned *dst, const bool a){return atomicAnd(dst,(unsigned int)a);}
  __device__ static bool ident_value() { return true; }

};

template <typename Number>
bool MultiGpuVector<Number>::all_zero () const
{
  std::vector<unsigned int *> res_d(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(unsigned int)));
    CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 1, sizeof(unsigned int )));

    const int nblocks = 1 + (local_size[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
    single_vec_reduction<AllZero<Number>> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],
                                                                                  local_size[i]);
    CUDA_CHECK_LAST;
  }

  std::vector<unsigned int> res(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(&res[i],res_d[i],sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));

    CUDA_CHECK_SUCCESS(cudaFree(res_d[i]));
  }

  bool global_res = true;
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    global_res = global_res && (res[i]!=0);
  }

  return global_res;
}

// scalar product

template <typename Number>
struct DotProd {
  typedef Number shmem_type;
  typedef Number result_type;
  __device__ static Number binary_op(const Number v1,const Number v2) {return v1*v2;}
  __device__ static Number red_op(const Number a, const Number b){return a+b;}
  __device__ static Number atomic_op(Number *dst, const Number a){return atomicAddWrapper(dst,(Number)a);}
  __device__ static Number ident_value() { return 0; }
};

template <typename Number>
Number MultiGpuVector<Number>::operator * (const MultiGpuVector<Number> &v) const
{
  AssertDimension(global_size, v.size());
  // FIXME: also check for compatible partitioners

  std::vector<Number *> res_d(partitioner->n_partitions());
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 0, sizeof(Number)));

    const int nblocks = 1 + (local_size[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);

    dual_vec_reduction<DotProd<Number> > <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],
                                                                                 v.vec[i],
                                                                                 local_size[i]);
    CUDA_CHECK_LAST;
  }

  std::vector<Number> res(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(&res[i],res_d[i],sizeof(Number),
                                  cudaMemcpyDeviceToHost));

    CUDA_CHECK_SUCCESS(cudaFree(res_d[i]));
  }

  Number global_res = 0;
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    global_res += res[i];
  }

  return global_res;
}


// Combined scaled addition of vector x into
// the current object and subsequent inner
// product of the current object with v
//
// (this + a*x).dot(v)

template <typename Number>
__global__ void add_and_dot_kernel(Number *res, Number *v1, const Number *v2,
                                   const Number *v3, const Number a, const int N)
{
  __shared__ Number res_buf[VR_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*VR_CHUNK_SIZE);
  const int local_idx = threadIdx.x;
  if(global_idx < N) {

    v1[global_idx]+= a*v2[global_idx];
    res_buf[local_idx] = (v1[global_idx])*v3[global_idx];
  }
  else
    res_buf[local_idx] = 0;

  for(int c = 1; c < VR_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*VR_BKSIZE;
    if(idx < N) {
      v1[idx] += a*v2[idx];
      res_buf[local_idx] += (v1[idx])*v3[idx];
    }
  }

  __syncthreads();

  reduce<DotProd<Number>,Number> (res,res_buf,local_idx,global_idx,N);
}

template <typename Number>
Number MultiGpuVector<Number>::add_and_dot (const Number  a,
                                       const MultiGpuVector<Number> &x,
                                       const MultiGpuVector<Number> &v)
{
  AssertDimension(global_size, x.size());
  AssertDimension(global_size, v.size());
  // FIXME: also check for compatible partitioners

  std::vector<Number *> res_d(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(Number)));
    CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 0, sizeof(Number)));

    const int nblocks = 1 + (local_size[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
    add_and_dot_kernel<Number> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],x.vec[i],
                                                                       v.vec[i],a,
                                                                       local_size[i]);
    CUDA_CHECK_LAST;
  }

  std::vector<Number> res(partitioner->n_partitions());

  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    CUDA_CHECK_SUCCESS(cudaMemcpy(&res[i],res_d[i],sizeof(Number),
                                  cudaMemcpyDeviceToHost));

    CUDA_CHECK_SUCCESS(cudaFree(res_d[i]));
  }

  Number global_res = 0;
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    global_res += res[i];
  }

  return global_res;
}

// needed kernels
template <typename Number>
__global__ void copy_with_indices_kernel(Number *dst, const Number *src,
                                         const unsigned int *src_indices,
                                         unsigned int n);
template <typename Number>
__global__ void add_with_indices_kernel(Number *dst,
                                        const unsigned int *dst_indices,
                                        const Number *src,
                                        unsigned int n);


template <typename Number>
void MultiGpuVector<Number>::compress(VectorOperation::values operation)
{
  // Assert(!vector_is_compressed,);

  // if(operation == VectorOperation::insert) {
  //   vector_is_compressed = true;
  //   return;
  // }

  // copy ghosted values from ghost_dofs section on other devices into
  // import_data on 'this' device
  for(int to=0; to<partitioner->n_partitions(); ++to) {
    // CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    for(int from=0; from<partitioner->n_partitions(); ++from) {
      if(to != from) {

        CUDA_CHECK_SUCCESS(cudaMemcpyPeer(import_data[to]+partitioner->import_data_offset(to,from),
                                          to,
                                          vec[from]+local_size[from]+partitioner->ghost_dofs_offset(from,to),
                                          from,
                                          partitioner->n_ghost_dofs(from,to)));

      }
    }
  }

  // on this device, merge stuff from import_data into the indices in
  // import_indices
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));

    const unsigned int n = partitioner->n_import_indices(i);
    const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

    add_with_indices_kernel<<<nblocks,COPY_WITH_INDEX_BKSIZE>>>(vec[i],
                                                                import_indices[i],
                                                                import_data[i], n);
    CUDA_CHECK_LAST;
  }

  vector_is_compressed = true;
}

template <typename Number>
void MultiGpuVector<Number>::update_ghost_values () const
{

  // Assert(vector_is_compressed);
  // Assert(!vector_is_ghosted);

  // on this device, copy stuff into import_data from the indices in
  // import_indices
  for(int i=0; i<partitioner->n_partitions(); ++i) {
    CUDA_CHECK_SUCCESS(cudaSetDevice(i));

    const unsigned int n = partitioner->n_import_indices(i);
    const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

    copy_with_indices_kernel<<<nblocks,COPY_WITH_INDEX_BKSIZE >>>(import_data[i],vec[i],
                                                                  import_indices[i],n);
    CUDA_CHECK_LAST;
  }

  // copy ghosted values from import_data on device `from` into
  // ghost_dofs section on device `to`

  for(int to=0; to<partitioner->n_partitions(); ++to) {
    // CUDA_CHECK_SUCCESS(cudaSetDevice(i));
    for(int from=0; from<partitioner->n_partitions(); ++from) {
      if(to != from) {
        CUDA_CHECK_SUCCESS(cudaMemcpyPeer(vec[to]+local_size[to]+partitioner->ghost_dofs_offset(to,from),
                                          to,
                                          import_data[from]+partitioner->import_data_offset(from,to),
                                          from,
                                          partitioner->n_ghost_dofs(to,from)));

      }
    }
  }

  vector_is_ghosted = true;
}


//=============================================================================
// non-class functions
//=============================================================================

template <typename Number>
__global__ void copy_with_indices_kernel(Number *dst, const Number *src,
                                         const unsigned int *src_indices,
                                         unsigned int n)
{
  const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    dst[i] = src[src_indices[i]];
  }
}

template <typename Number>
__global__ void add_with_indices_kernel(Number *dst,
                                        const unsigned int *dst_indices,
                                        const Number *src,
                                        unsigned int n)
{
  const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    atomicAdd(&dst[dst_indices[i]],src[i]);
  }
}


//=============================================================================
// instantiate templates
//=============================================================================

template class MultiGpuVector<double>;
template class MultiGpuVector<float>;

template MultiGpuVector<float>::MultiGpuVector(const MultiGpuVector<double>&);
template MultiGpuVector<double>::MultiGpuVector(const MultiGpuVector<float>&);
template MultiGpuVector<float>& MultiGpuVector<float>::operator=(const MultiGpuVector<double>&);
template MultiGpuVector<double>& MultiGpuVector<double>::operator=(const MultiGpuVector<float>&);

// #define INSTANTIATE_COPY_WITH_INDICES(dst_type,src_type,idx_type)       \
//   template void copy_with_indices<dst_type,src_type,idx_type> (MultiGpuVector<dst_type> &, const MultiGpuVector<src_type> &, \
//                                                                const GpuList<idx_type> &, const GpuList<idx_type> &)
// INSTANTIATE_COPY_WITH_INDICES(double,double,int);
// INSTANTIATE_COPY_WITH_INDICES(double,double,unsigned int);

// INSTANTIATE_COPY_WITH_INDICES(float,float,int);
// INSTANTIATE_COPY_WITH_INDICES(float,float,unsigned int);

// INSTANTIATE_COPY_WITH_INDICES(double,float,int);
// INSTANTIATE_COPY_WITH_INDICES(double,float,unsigned int);

// INSTANTIATE_COPY_WITH_INDICES(float,double,int);
// INSTANTIATE_COPY_WITH_INDICES(float,double,unsigned int);
