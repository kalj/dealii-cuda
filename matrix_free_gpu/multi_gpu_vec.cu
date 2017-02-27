/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)multi_gpu_vec.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */


#include "multi_gpu_vec.h"
#include "atomic.cuh"
#include "cuda_utils.cuh"
#include "cuda_memory_utils.h"

#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8
#define VR_BKSIZE 512
#define VR_CHUNK_SIZE 8
#define COPY_WITH_INDEX_BKSIZE 256



namespace dealii
{



  //=============================================================================
  // Constructors / assignment
  //=============================================================================

  template <typename Number>
  MultiGpuVector<Number>::DevRef& MultiGpuVector<Number>::DevRef::operator=(const Number value)
  {
    CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(owning_device)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(ptr,&value,sizeof(Number),
                                  cudaMemcpyHostToDevice));
    return *this;
  }

  template <typename Number>
  MultiGpuVector<Number>::MultiGpuVector(const std::shared_ptr<const GpuPartitioner> &partitioner_in)
    : vec(partitioner_in->n_partitions()),
      import_data(partitioner,partitioner->n_import_indices()),
      import_indices(partitioner,partitioner->n_import_indices()),
      local_sizes(partitioner_in->n_partitions()),
      global_size(partitioner_in->n_global_dofs()),
      partitioner(partitioner_in),
      vector_is_ghosted(false),
      vector_is_compressed(true)
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      local_sizes[i] = partitioner->n_dofs(i);
      const unsigned int ghosted_size = local_sizes[i] + partitioner->n_ghost_dofs_tot(i);
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
    }

    import_indices = partitioner->import_indices();
  }

  template <typename Number>
  MultiGpuVector<Number>::MultiGpuVector(const MultiGpuVector<Number>& old)
    : vec(old.partitioner->n_partitions()),
      import_data(old.import_data),
      import_indices(old.import_indices),
      local_sizes(old.partitioner->n_partitions()),
      global_size(old.partitioner->n_global_dofs()),
      partitioner(old.partitioner),
      vector_is_ghosted(old.vector_is_ghosted),
      vector_is_compressed(old.vector_is_compressed)
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      local_sizes[i] = partitioner->n_dofs(i);
      const unsigned int ghosted_size = local_sizes[i] + partitioner->n_ghost_dofs_tot(i);
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],old.vec[i],ghosted_size*sizeof(Number),
                                    cudaMemcpyDeviceToDevice));
    }
  }


  // copy constructor from vector based on other number type
  template <typename Number>
  template <typename OtherNumber>
  MultiGpuVector<Number>::MultiGpuVector(const MultiGpuVector<OtherNumber>& old)
    : vec(old.partitioner->n_partitions()),
      import_data(old.import_data),
      import_indices(old.import_indices),
      local_sizes(old.partitioner->n_partitions()),
      global_size(old.partitioner->n_global_dofs()),
      partitioner(old.partitioner),
      vector_is_ghosted(old.vector_is_ghosted),
      vector_is_compressed(old.vector_is_compressed)
  {

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      local_sizes[i] = partitioner->n_dofs(i);
      const unsigned int ghosted_size = local_sizes[i] + partitioner->n_ghost_dofs_tot(i);

      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],ghosted_size*sizeof(Number)));
      internal::copy_dev_array(vec[i],old.vec[i],ghosted_size);
    }
  }

  template <typename Number>
  MultiGpuVector<Number>::~MultiGpuVector()
  {
    for(int i=0; i<vec.size(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      if(vec[i] != NULL) {
        CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
      }
    }
  }



  template <typename Number>
  MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const MultiGpuVector<Number>& old)
  {
    AssertDimension(global_size, old.size());
    // FIXME: also check for compatible partitioners

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      const unsigned int ghosted_size = local_sizes[i] + partitioner->n_ghost_dofs_tot(i);
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                    old.vec[i],
                                    ghosted_size*sizeof(Number),
                                    cudaMemcpyDeviceToDevice));
    }

    vector_is_ghosted = old.vector_is_ghosted;
    vector_is_ghosted = old.vector_is_compressed;

    return *this;
  }


  // same for assignment
  template <typename Number>
  template <typename OtherNumber>
  MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const MultiGpuVector<OtherNumber>& old)
  {
    AssertDimension(global_size, old.size());
    // FIXME: also check for compatible partitioners

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      const unsigned int ghosted_size = local_sizes[i] + partitioner->n_ghost_dofs_tot(i);
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      internal::copy_dev_array(vec[i],old.vec[i],ghosted_size);
    }

    vector_is_ghosted = old.vector_is_ghosted;
    vector_is_compressed = old.vector_is_compressed;

    return *this;
  }

      template <typename Number>
  MultiGpuVector<Number>& MultiGpuVector<Number>::operator=(const Vector<Number>& old_cpu)
  {
    AssertDimension(global_size, old_cpu.size());

    const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                    cpu_data+partitioner->local_dof_offset(i),
                                    local_sizes[i]*sizeof(Number),
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(vec[i],
                                    cpu_data+partitioner->local_dof_offset(i),
                                    local_sizes[i]*sizeof(Number),
                                    cudaMemcpyHostToDevice));
    }

    vector_is_ghosted = false;
    vector_is_compressed = true;

    return *this;
  }


  template <typename Number>
  void MultiGpuVector<Number>::copyToHost(Vector<Number>& dst) const
  {
    AssertDimension(global_size, dst.size());

    // FIXME: probably check that this vector is compressed

    Number *cpu_data = &dst[0];

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(cpu_data+partitioner->local_dof_offset(i),
                                    vec[i],
                                    local_sizes[i]*sizeof(Number),
                                    cudaMemcpyDeviceToHost));
    }
  }


  template <typename Number>
  void MultiGpuVector<Number>::reinit (unsigned int s)
  {
    AssertThrow(s==0,ExcMessage("cannot change size of DoFVector, except for size 0"));

    if(partitioner != NULL)
      for(int i=0; i<partitioner->n_partitions(); ++i) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        if(vec[i] != NULL) {
          CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
        }
      }

    vec.resize(0);

    partitioner = NULL;
  }


  // initialize with a partitioner
  template <typename Number>
  void MultiGpuVector<Number>::reinit (const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                                       bool leave_elements_uninitialized)
  {
    if(partitioner != NULL) {
      // we have a partitioner

      // if we already have same or equivalent partitioner, just return
      // FIXME: this doesn't work now, as our partitioner has probably changed....
      // if(partitioner->is_compatible(*partitioner_in))
        // return;

      if( partitioner_in->n_partitions() > partitioner->n_partitions()) {
        local_sizes.resize(partitioner_in->n_partitions(),0);
        vec.resize(partitioner_in->n_partitions(), NULL);
      }
      else if(partitioner_in->n_partitions() < partitioner->n_partitions()) {
        local_sizes.resize(partitioner_in->n_partitions());
        // free up items to remove
        for(int i=partitioner_in->n_partitions(); i<partitioner->n_partitions(); ++i) {
          if(vec[i] != NULL) {
            CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
            CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
            vec[i] = NULL;
          }
        }
        vec.resize(partitioner_in->n_partitions());
      }
    }
    else {
      local_sizes.resize(partitioner_in->n_partitions(),0);
      vec.resize(partitioner_in->n_partitions(),NULL);
    }

    // now we have a vec of the right length, but with possibly unallocated
    // and misformed storage

    for(int i=0; i<partitioner_in->n_partitions(); ++i) {

      if(local_sizes[i] != partitioner_in->n_dofs(i)) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        if(vec[i] != NULL) {
          CUDA_CHECK_SUCCESS(cudaFree(vec[i]));
        }
        CUDA_CHECK_SUCCESS(cudaMalloc(&vec[i],partitioner_in->n_dofs(i)*sizeof(Number)));
        local_sizes[i] = partitioner_in->n_dofs(i);
      }
    }

    partitioner = partitioner_in;
    global_size = partitioner_in->n_global_dofs();

    // now just need to set up ghost data
    import_data.reinit(partitioner_in,partitioner->n_import_indices());
    import_indices.reinit(partitioner_in,partitioner->n_import_indices());
    import_indices = partitioner_in->import_indices();

    vector_is_ghosted = false;
    vector_is_compressed = true;

    if(!leave_elements_uninitialized)
      for(int i=0; i<partitioner->n_partitions(); ++i) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        CUDA_CHECK_SUCCESS(cudaMemset(vec[i], 0, local_sizes[i]*sizeof(Number)));
      }

  }


  // resize to have the same structure as the one provided and/or clear
  // vector. note that the second argument must have a default value equal to
  // false
  template <typename Number>
  void MultiGpuVector<Number>::reinit (const MultiGpuVector<Number>& other,
                                       bool leave_elements_uninitialized)
  {
    reinit(other.partitioner,leave_elements_uninitialized);
  }


  template <typename Number>
  Number MultiGpuVector<Number>::operator()(const size_t i) const
  {
    // FIXME: probably check for compressed

    const int owning_device = partitioner->dof_owner(i);
    const int local_index = partitioner->local_index(owning_device,i);

    Number value;

    CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
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
    const int local_index = partitioner->local_index(owning_device,i);
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_sadd<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],a,b,local_sizes[i]);
      CUDA_CHECK_LAST;
    }
  }



  // element-wise multiplication of vectors
  template <typename Number>
  void MultiGpuVector<Number>::scale (const MultiGpuVector<Number> &x)
  {
    AssertDimension(global_size, x.size());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_bin_op<Number,Binop_Multiplication> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],
                                                                               local_sizes[i]);
      CUDA_CHECK_LAST;
    }
  }

  // element-wise division of vectors
  template <typename Number>
  MultiGpuVector<Number>& MultiGpuVector<Number>::operator/= (const MultiGpuVector<Number> &x)
  {
    AssertDimension(global_size, x.size());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_bin_op<Number,Binop_Division> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],
                                                                         local_sizes[i]);
      CUDA_CHECK_LAST;
    }
    return *this;
  }

  template <typename Number>
  MultiGpuVector<Number>& MultiGpuVector<Number>::invert()
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_invert<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],local_sizes[i]);
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_equ<Number,Number,Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],x.vec[i],a,
                                                                     local_sizes[i]);
      CUDA_CHECK_LAST;
    }
  }


  // scale the elements of the vector
  // by a fixed value
  template <typename Number>
  MultiGpuVector<Number> & MultiGpuVector<Number>::operator *= (const Number a)
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_scale<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],a,local_sizes[i]);
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      const int nblocks = 1 + (local_sizes[i]-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      vec_init<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec[i],a,local_sizes[i]);
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(unsigned int)));
      CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 1, sizeof(unsigned int )));

      const int nblocks = 1 + (local_sizes[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
      single_vec_reduction<AllZero<Number>> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],
                                                                                    local_sizes[i]);
      CUDA_CHECK_LAST;
    }

    std::vector<unsigned int> res(partitioner->n_partitions());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(Number)));
      CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 0, sizeof(Number)));

      const int nblocks = 1 + (local_sizes[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);

      dual_vec_reduction<DotProd<Number> > <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],
                                                                                   v.vec[i],
                                                                                   local_sizes[i]);
      CUDA_CHECK_LAST;
    }

    std::vector<Number> res(partitioner->n_partitions());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
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
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&res_d[i],sizeof(Number)));
      CUDA_CHECK_SUCCESS(cudaMemset(res_d[i], 0, sizeof(Number)));

      const int nblocks = 1 + (local_sizes[i]-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
      add_and_dot_kernel<Number> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d[i],vec[i],x.vec[i],
                                                                         v.vec[i],a,
                                                                         local_sizes[i]);
      CUDA_CHECK_LAST;
    }

    std::vector<Number> res(partitioner->n_partitions());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
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
      // CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      for(int from=0; from<partitioner->n_partitions(); ++from) {
        if(to != from) {

#ifndef FAKE_MULTI_GPU
          CUDA_CHECK_SUCCESS(cudaMemcpyPeer(import_data.getData(to)+partitioner->import_data_offset(to,from),
                                            partitioner->get_partition_id(to),
                                            vec[from]+local_sizes[from]+partitioner->ghost_dofs_offset(from,to),
                                            partitioner->get_partition_id(from),
                                            partitioner->n_ghost_dofs(from,to)*sizeof(Number)));
#else
          CUDA_CHECK_SUCCESS(cudaMemcpy(import_data.getData(to)+partitioner->import_data_offset(to,from),
                                        vec[from]+local_sizes[from]+partitioner->ghost_dofs_offset(from,to),
                                        partitioner->n_ghost_dofs(from,to)*sizeof(Number),
                                        cudaMemcpyDeviceToDevice));
#endif

        }
      }
    }

    // on this device, merge stuff from import_data into the indices in
    // import_indices

    add_with_indices<true>(*this,import_indices,import_data);

    vector_is_compressed = true;
  }


  template <typename Number>
  void MultiGpuVector<Number>::update_ghost_values () const
  {

    // Assert(vector_is_compressed);
    // Assert(!vector_is_ghosted);

    // on this device, copy stuff into import_data from the indices in
    // import_indices
    copy_with_indices(import_data,*this,import_indices);


    // copy ghosted values from import_data on device `from` into
    // ghost_dofs section on device `to`
    for(int to=0; to<partitioner->n_partitions(); ++to) {
      // CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      for(int from=0; from<partitioner->n_partitions(); ++from) {
        if(to != from) {

#ifndef FAKE_MULTI_GPU
          CUDA_CHECK_SUCCESS(cudaMemcpyPeer(vec[to]+local_sizes[to]+partitioner->ghost_dofs_offset(to,from),
                                            partitioner->get_partition_id(to),
                                            import_data.getDataRO(from)+partitioner->import_data_offset(from,to),
                                            partitioner->get_partition_id(from),
                                            partitioner->n_ghost_dofs(to,from)*sizeof(Number)));
#else
          CUDA_CHECK_SUCCESS(cudaMemcpy(vec[to]+local_sizes[to]+partitioner->ghost_dofs_offset(to,from),
                                        import_data.getDataRO(from)+partitioner->import_data_offset(from,to),
                                        partitioner->n_ghost_dofs(to,from)*sizeof(Number),
                                        cudaMemcpyDeviceToDevice));
#endif
        }
      }
    }

    vector_is_ghosted = true;
  }


  //=============================================================================
  // non-class functions
  //=============================================================================

  namespace kernels
  {
    template <typename DstNumber, typename SrcNumber, typename IndexT>
    __global__ void copy_with_indices(DstNumber *dst, const IndexT *dst_indices,
                                      const SrcNumber *src,
                                      const IndexT *src_indices, int n);

    template <typename Number>
    __global__ void copy_with_indices(Number *dst, const Number *src,
                                      const unsigned int *src_indices,
                                      unsigned int n);

    template <typename Number>
    __global__ void copy_with_indices(Number *dst,
                                      const unsigned int *dst_indices,
                                      const Number *src,
                                      unsigned int n);


    template <bool atomic, typename Number>
    __global__ void add_with_indices(Number *dst,
                                     const unsigned int *dst_indices,
                                     const Number *src,
                                     unsigned int n);
  }




  template <typename DstNumber, typename SrcNumber>
  void copy_with_indices(MultiGpuVector<DstNumber> &dst, const MultiGpuList<unsigned int> &dst_indices,
                         const MultiGpuVector<SrcNumber> &src, const MultiGpuList<unsigned int> &src_indices)
  {
    for(int i=0; i< dst.get_partitioner()->n_partitions(); ++i) {

      const unsigned int n = dst_indices.local_size(i);
      const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

      if(n>0) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        kernels::copy_with_indices<<<nblocks,COPY_WITH_INDEX_BKSIZE>>>(dst.getData(i),
                                                                       dst_indices.getDataRO(i),
                                                                       src.getDataRO(i),
                                                                       src_indices.getDataRO(i),
                                                                       n);
        CUDA_CHECK_LAST;
      }
    }
  }


  template <typename Number>
  void copy_with_indices(MultiGpuVector<Number> &dst,
                         const MultiGpuList<unsigned int> &dst_indices,
                         const MultiGpuList<Number> &src)
  {
    for(int i=0; i< dst.get_partitioner()->n_partitions(); ++i) {

      const unsigned int n = dst_indices.local_size(i);
      const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

      if(n>0) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        kernels::copy_with_indices<<<nblocks,COPY_WITH_INDEX_BKSIZE>>>(dst.getData(i),
                                                                       dst_indices.getDataRO(i),
                                                                       src.getDataRO(i), n);
        CUDA_CHECK_LAST;
      }
    }
  }


  template <typename Number>
  void copy_with_indices(MultiGpuList<Number> &dst,
                         const MultiGpuVector<Number> &src,
                         const MultiGpuList<unsigned int> &src_indices)
  {
    for(int i=0; i< src.get_partitioner()->n_partitions(); ++i) {

      const unsigned int n = src_indices.local_size(i);
      const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

      if(n>0) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        kernels::copy_with_indices<<<nblocks,COPY_WITH_INDEX_BKSIZE>>>(dst.getData(i),
                                                                       src.getDataRO(i),
                                                                       src_indices.getDataRO(i), n);
        CUDA_CHECK_LAST;
      }
    }
  }

  template <bool atomic, typename Number>
  void add_with_indices(MultiGpuVector<Number> &dst,
                        const MultiGpuList<unsigned int> &dst_indices,
                        const MultiGpuList<Number> &src)

  {
    for(int i=0; i< dst.get_partitioner()->n_partitions(); ++i) {

      const unsigned int n = dst_indices.local_size(i);
      const unsigned int nblocks = 1 + (n-1) / COPY_WITH_INDEX_BKSIZE;

      if(n>0) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        kernels::add_with_indices<atomic><<<nblocks,COPY_WITH_INDEX_BKSIZE>>>(dst.getData(i),
                                                                              dst_indices.getDataRO(i),
                                                                              src.getDataRO(i), n);
        CUDA_CHECK_LAST;
      }
    }
  }

  namespace kernels
  {
    template <typename DstNumber, typename SrcNumber, typename IndexT>
    __global__ void copy_with_indices(DstNumber *dst, const IndexT *dst_indices, const SrcNumber *src,
                                      const IndexT *src_indices, int n)
    {
      const int i = threadIdx.x + blockIdx.x*blockDim.x;
      if(i<n) {
        dst[dst_indices[i]] = src[src_indices[i]];
      }
    }


    template <typename Number>
    __global__ void copy_with_indices(Number *dst, const Number *src,
                                      const unsigned int *src_indices,
                                      unsigned int n)
    {
      const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
      if(i<n) {
        dst[i] = src[src_indices[i]];
      }
    }

    template <typename Number>
    __global__ void copy_with_indices(Number *dst,
                                      const unsigned int *dst_indices,
                                      const Number *src,
                                      unsigned int n)
    {
      const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
      if(i<n) {
        dst[dst_indices[i]] = src[i];
      }
    }

    template <bool atomic, typename Number>
    __global__ void add_with_indices(Number *dst,
                                     const unsigned int *dst_indices,
                                     const Number *src,
                                     unsigned int n)
    {
      const unsigned int i = threadIdx.x + blockIdx.x*blockDim.x;
      if(i<n) {
        if(atomic)
          atomicAddWrapper(&dst[dst_indices[i]],src[i]);
        else
          dst[dst_indices[i]] += src[i];
      }
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

  template void add_with_indices<false,double>(MultiGpuVector<double> &,
                                               const MultiGpuList<unsigned int> &,
                                               const MultiGpuList<double> &);
  template void add_with_indices<false,float>(MultiGpuVector<float> &,
                                              const MultiGpuList<unsigned int> &,
                                              const MultiGpuList<float> &);
  template void add_with_indices<true,double>(MultiGpuVector<double> &,
                                              const MultiGpuList<unsigned int> &,
                                              const MultiGpuList<double> &);
  template void add_with_indices<true,float>(MultiGpuVector<float> &,
                                             const MultiGpuList<unsigned int> &,
                                             const MultiGpuList<float> &);

  template void copy_with_indices<double>(MultiGpuList<double> &,
                                          const MultiGpuVector<double> &,
                                          const MultiGpuList<unsigned int> &);
  template void copy_with_indices<float>(MultiGpuList<float> &,
                                         const MultiGpuVector<float> &,
                                         const MultiGpuList<unsigned int> &);

  template void copy_with_indices<double>(MultiGpuVector<double> &,
                                          const MultiGpuList<unsigned int> &,
                                          const MultiGpuList<double> &);
  template void copy_with_indices<float>(MultiGpuVector<float> &,
                                         const MultiGpuList<unsigned int> &,
                                         const MultiGpuList<float> &);


#define INSTANTIATE_COPY_WITH_INDICES(dst_type,src_type)                \
  template void copy_with_indices<dst_type,src_type> (MultiGpuVector<dst_type> &, const MultiGpuList<unsigned int> &, \
                                                      const MultiGpuVector<src_type> &, const MultiGpuList<unsigned int> &)

  INSTANTIATE_COPY_WITH_INDICES(double,double);

  INSTANTIATE_COPY_WITH_INDICES(float,float);

  INSTANTIATE_COPY_WITH_INDICES(double,float);

  INSTANTIATE_COPY_WITH_INDICES(float,double);

}

// explicitly instantiate vector memories for this
#include <deal.II/lac/vector_memory.templates.h>

namespace dealii
{
  template class VectorMemory<MultiGpuVector<double> >;
  template class GrowingVectorMemory<MultiGpuVector<double> >;
  template class VectorMemory<MultiGpuVector<float> >;
  template class GrowingVectorMemory<MultiGpuVector<float> >;
}
