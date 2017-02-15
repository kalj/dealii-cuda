/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)gpu_vec.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#include <cfloat>

#include "gpu_vec.h"
#include "atomic.cuh"
#include "cuda_utils.cuh"

#include <deal.II/lac/vector_memory.templates.h>
template class VectorMemory<GpuVector<double> >;
template class GrowingVectorMemory<GpuVector<double> >;
template class VectorMemory<GpuVector<float> >;
template class GrowingVectorMemory<GpuVector<float> >;


//=============================================================================
// Constructors / assignment
//=============================================================================

template <typename Number>
GpuVector<Number>::DevRef& GpuVector<Number>::DevRef::operator=(const Number value)
{
  CUDA_CHECK_SUCCESS(cudaMemcpy(ptr,&value,sizeof(Number),
                                cudaMemcpyHostToDevice));
  return *this;
}

template <typename Number>
GpuVector<Number>::GpuVector(unsigned int s)
: _size(s) {
  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,s*sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemset(vec_dev,0,s*sizeof(Number)));
}

template <typename Number>
GpuVector<Number>::GpuVector(const GpuVector<Number>& old)
  : _size(old.size()) {
  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,old.vec_dev,_size*sizeof(Number),
                                cudaMemcpyDeviceToDevice));
}


template <typename Number>
GpuVector<Number>::GpuVector(const Vector<Number>& old_cpu)
  : _size(old_cpu.size()) {
  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,cpu_data,_size*sizeof(Number),
                                cudaMemcpyHostToDevice));
}

template <typename Number>
GpuVector<Number>::GpuVector(const std::vector<Number>& old_cpu)
  : _size(old_cpu.size()) {
  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  // const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];
  const Number *cpu_data = old_cpu.data();
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,cpu_data,_size*sizeof(Number),
                                cudaMemcpyHostToDevice));
}

template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator=(const Vector<Number>& old_cpu) {
  if(_size != old_cpu.size()) {
    if(vec_dev != NULL)
      CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
    _size = old_cpu.size();
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  }
  const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,cpu_data,_size*sizeof(Number),
                                cudaMemcpyHostToDevice));
  return *this;
}

template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator=(const std::vector<Number>& old_cpu) {
  if(_size != old_cpu.size()) {
    if(vec_dev != NULL)
      CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
    _size = old_cpu.size();
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  }
  // const Number *cpu_data = &const_cast<Vector<Number>&>(old_cpu)[0];
  const Number *cpu_data = old_cpu.data();
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,cpu_data,_size*sizeof(Number),
                                cudaMemcpyHostToDevice));
  return *this;
}

template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator=(const GpuVector<Number>& old) {
  if(_size != old._size) {
    if(vec_dev != NULL) CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
    _size = old._size;
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));
  }
  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,old.vec_dev,_size*sizeof(Number),
                                cudaMemcpyDeviceToDevice));
  return *this;
}


template <typename Number>
GpuVector<Number>::~GpuVector() {
  if(vec_dev != NULL) {
    CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
  }
}

template <typename Number>
void GpuVector<Number>::copyToHost(Vector<Number>& dst) const {
  CUDA_CHECK_SUCCESS(cudaMemcpy(&dst[0],getDataRO(),_size*sizeof(Number),
                                cudaMemcpyDeviceToHost));
}

template <typename Number>
void GpuVector<Number>::fromHost(const Number *buf, unsigned int n)
{
  /* FIXME: better dimension check */
  if(n != _size) {
    fprintf(stderr,"ERROR: Non-matching dimensions!\n");
    return;
  }

  cudaMemcpy(vec_dev,buf,_size*sizeof(Number),
             cudaMemcpyHostToDevice);
  cudaAssertNoError();
}



// resize to have the same structure as the one provided and/or clear
// vector. note that the second argument must have a default value equal to
// false
template <typename Number>
void GpuVector<Number>::reinit (const GpuVector<Number>& other,
                                bool leave_elements_uninitialized)
{
  if(other._size != this->_size) {
    if(vec_dev != NULL) {
      CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
    }

    _size = other._size;
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));

  }
  if(!leave_elements_uninitialized) {
    CUDA_CHECK_SUCCESS(cudaMemset(vec_dev, 0, _size*sizeof(Number)));
  }
}

template <typename Number>
void GpuVector<Number>::resize (unsigned int n)
{
  if(n != this->_size) {
    if(vec_dev != NULL) {
      CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
    }

    _size = n;
    CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,_size*sizeof(Number)));

  }
}


template <typename Number>
Number GpuVector<Number>::operator()(const size_t i) const
{
  Number value;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&value,vec_dev+i,sizeof(Number),
                                cudaMemcpyDeviceToHost));
  return value;
}

// necessary for deal.ii but shouldn't be used!
template <typename Number>
GpuVector<Number>::DevRef GpuVector<Number>::operator()(const size_t i)
{
  return GpuVector<Number>::DevRef(vec_dev+i);
}


//=============================================================================
// Element wise operations (mult. with scalar, vector addition)
//=============================================================================

#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8


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


template <typename Number>
__global__ void vec_equ(Number *v1, const Number *v2, const Number a, const int N)
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
void GpuVector<Number>::sadd (const Number a,
                              const Number b,
                              const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_sadd<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,a,b,_size);
    CUDA_CHECK_LAST;
}



// element-wise multiplication of vectors
template <typename Number>
void GpuVector<Number>::scale (const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_bin_op<Number,Binop_Multiplication> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,_size);
    CUDA_CHECK_LAST;
}

// element-wise division of vectors
template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator/= (const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_bin_op<Number,Binop_Division> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,_size);
    CUDA_CHECK_LAST;
  return *this;
}

template <typename Number>
GpuVector<Number>& GpuVector<Number>::invert()
{
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_invert<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,_size);
    CUDA_CHECK_LAST;
  return *this;
}

// scaled assignment of a vector
template <typename Number>
void GpuVector<Number>::equ (const Number a,
                             const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_equ<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,a,_size);
  CUDA_CHECK_LAST;
}


// scale the elements of the vector
// by a fixed value
template <typename Number>
GpuVector<Number> & GpuVector<Number>::operator *= (const Number a) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_scale<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,a,_size);
    CUDA_CHECK_LAST;
  return *this;
}


// return the l2 norm of the vector
template <typename Number>
Number GpuVector<Number>::l2_norm () const {
  return sqrt((*this)*(*this));
}


// initialize vector with value
template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator=(const Number a) {

  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_init<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,a,_size);
    CUDA_CHECK_LAST;

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

#define VR_BKSIZE 512
#define VR_CHUNK_SIZE 8

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
bool GpuVector<Number>::all_zero () const {
  unsigned int *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(unsigned int)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 1, sizeof(unsigned int )));

  const int nblocks = 1 + (_size-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
  single_vec_reduction<AllZero<Number>> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d,vec_dev,_size);
    CUDA_CHECK_LAST;

  unsigned int res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(unsigned int),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));
  return !(res==0);
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
Number GpuVector<Number>::operator * (const GpuVector<Number> &v) const {
  Number *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 0, sizeof(Number)));

  const int nblocks = 1 + (_size-1) / (VR_CHUNK_SIZE*VR_BKSIZE);

  dual_vec_reduction<DotProd<Number> > <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d,vec_dev,
                                                                               v.vec_dev,_size);
    CUDA_CHECK_LAST;
  Number res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(Number),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));
  return res;
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
Number GpuVector<Number>::add_and_dot (const Number  a,
                                       const GpuVector<Number> &x,
                                       const GpuVector<Number> &v) {

  Number *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 0, sizeof(Number)));

  const int nblocks = 1 + (_size-1) / (VR_CHUNK_SIZE*VR_BKSIZE);
  add_and_dot_kernel<Number> <<<dim3(nblocks,1),dim3(VR_BKSIZE,1)>>>(res_d,vec_dev,x.vec_dev,
                                                                     v.vec_dev,a,_size);
    CUDA_CHECK_LAST;

  Number res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(Number),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));

  return res;
}



//=============================================================================
// non-class functions
//=============================================================================
template <typename Number, typename IndexT>
__global__ void copy_with_indices_kernel(Number *dst, const Number *src, const IndexT *dst_indices, const IndexT *src_indices, int n)
{
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    dst[dst_indices[i]] = src[src_indices[i]];
  }
}


template <typename Number, typename IndexT>
void copy_with_indices(GpuVector<Number> &dst, const GpuVector<Number> &src,
                       const GpuList<IndexT> &dst_indices, const GpuList<IndexT> &src_indices)
{
  const int n = dst_indices.size();
  const int blocksize = 256;
  const dim3 block_dim = dim3(blocksize);
  const dim3 grid_dim = dim3(1 + (n-1)/blocksize);
  copy_with_indices_kernel<<<grid_dim, block_dim >>>(dst.getData(),src.getDataRO(),dst_indices.getDataRO(),src_indices.getDataRO(),n);
    CUDA_CHECK_LAST;
}


//=============================================================================
// instantiate templates
//=============================================================================

template class GpuVector<double>;
template class GpuVector<float>;

template void copy_with_indices<double> (GpuVector<double> &, const GpuVector<double> &,
                                         const GpuList<int> &, const GpuList<int> &);
template void copy_with_indices<double> (GpuVector<double> &, const GpuVector<double> &,
                                         const GpuList<unsigned int> &, const GpuList<unsigned int> &);

template void copy_with_indices<float> (GpuVector<float> &, const GpuVector<float> &,
                                         const GpuList<int> &, const GpuList<int> &);
template void copy_with_indices<float> (GpuVector<float> &, const GpuVector<float> &,
                                         const GpuList<unsigned int> &, const GpuList<unsigned int> &);
