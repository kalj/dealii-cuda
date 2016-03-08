/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)gpu_vec.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

// #include <cassert>
// #include <fstream>

#include <cfloat>

#include "gpu_vec.h"
#include "atomic.cuh"
#include "cuda_utils.cuh"

#include <deal.II/lac/vector_memory.templates.h>
template class VectorMemory<GpuVector<double> >;
template class GrowingVectorMemory<GpuVector<double> >;


// Constructors / assignment

template <typename Number>
GpuVector<Number>::GpuVector(unsigned int s)
: _size(s) {
  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,s*sizeof(Number)));
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

// cuda kernels
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


//=============================================================================
// Reduction operations
//=============================================================================


/*
  Performance serial reduction:
  bksize        time
  1024          1.03
  512           0.426
  256           0.433
  128           0.79
  64            1.76
  32            3.4
  16            6.8

  Performance parallel reduction, every s'th thread
  1024          0.25
  512           0.202
  256           0.48

  Performance parallel reduction, the bksize/s first threads
  1024          0.203
  512           0.206
  256           0.499

  Performance -||-, improved looping
  1024          0.2003

  Performance chunking (cs=2)
  1024          0.131
  512           0.124
  256           0.272

  Performance chunking (cs=4)
  1024          0.103
  512           0.085
  256           0.153

  Performance chunking (cs=8)
  1024          0.100
  512           0.083
  256           0.093

  Performance chunking (cs=16)
  1024          0.091
  512           0.084
  256           0.084

  Performance (cs=8), unroll from 32
  1024          0.097
  512           0.083
  256           0.092

*/

#ifndef DOT_BKSIZE
#define DOT_BKSIZE 512
#endif
#ifndef DOT_CHUNK_SIZE
#define DOT_CHUNK_SIZE 8
#endif

template <typename Number>
__device__ void dotWithinWarp(volatile Number *res_buf, int local_idx) {
  if(DOT_BKSIZE >= 64) res_buf[local_idx] += res_buf[local_idx+32];
  if(DOT_BKSIZE >= 32) res_buf[local_idx] += res_buf[local_idx+16];
  if(DOT_BKSIZE >= 16) res_buf[local_idx] += res_buf[local_idx+8];
  if(DOT_BKSIZE >= 8)  res_buf[local_idx] += res_buf[local_idx+4];
  if(DOT_BKSIZE >= 4)  res_buf[local_idx] += res_buf[local_idx+2];
  if(DOT_BKSIZE >= 2)  res_buf[local_idx] += res_buf[local_idx+1];
}

template <typename Number>
__global__ void dot_prod(Number *res, const Number *v1, const Number *v2, const int N)
{
  __shared__ Number res_buf[DOT_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*DOT_CHUNK_SIZE);
  const int local_idx = threadIdx.x;
  if(global_idx < N)
    res_buf[local_idx] = v1[global_idx]*v2[global_idx];
  else
    res_buf[local_idx] = 0;

  for(int c = 1; c < DOT_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*DOT_BKSIZE;
    if(idx < N)
      res_buf[local_idx] += v1[idx]*v2[idx];
  }

  __syncthreads();
  // if(local_idx == 0) {
  //     Number contrib = 0;
  //     for(int i = 0; i < DOT_BKSIZE; ++i) {
  //         contrib += res_buf[i];
  //     }
  //     atomicAdd(res,contrib);
  // }
  for(int s = DOT_BKSIZE/2; s>32; s=s>>1) {
    // for(int s = DOT_BKSIZE/2; s>0; s=s>>1) {
    // if(local_idx % s == 0)
    // res_buf[local_idx] += res_buf[local_idx+s/2];

    if(local_idx < s)
      res_buf[local_idx] += res_buf[local_idx+s];
    __syncthreads();
  }

  if(local_idx < 32) {
    dotWithinWarp(res_buf,local_idx);
  }

  if(local_idx == 0)
    atomicAdd(res,res_buf[0]);
}

// scalar product
template <typename Number>
Number GpuVector<Number>::operator * (const GpuVector<Number> &v) const {
  Number *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 0, sizeof(Number)));

  const int nblocks = 1 + (_size-1) / (DOT_CHUNK_SIZE*DOT_BKSIZE);
  dot_prod<Number> <<<dim3(nblocks,1),dim3(DOT_BKSIZE,1)>>>(res_d,vec_dev,v.vec_dev,_size);

  Number res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(Number),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));
  return res;
}


// single vector reduction, e.g. all_zero, max, min, etc

#ifndef SVR_BKSIZE
#define SVR_BKSIZE 512
#endif
#ifndef SVR_CHUNK_SIZE
#define SVR_CHUNK_SIZE 8
#endif


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
struct MaxOperation {
  typedef Number shmem_type;
  typedef Number result_type;
  __device__ static Number elemwise_op(const Number v) {return v == 0;}
  __device__ static Number red_op(const Number a, const Number b){return a>b?a:b;}
  __device__ static Number atomic_op(Number *dst, const Number a){return atomicMax(dst,a);}
  __device__ static Number ident_value() { return -DBL_MAX; }
};


template <typename Operation, typename Number>
__device__ void reduceWithinWarp(volatile typename Operation::shmem_type *res_buf, int local_idx) {
  if(DOT_BKSIZE >= 64) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+32]);
  if(DOT_BKSIZE >= 32) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+16]);
  if(DOT_BKSIZE >= 16) res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+8]);
  if(DOT_BKSIZE >= 8)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+4]);
  if(DOT_BKSIZE >= 4)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+2]);
  if(DOT_BKSIZE >= 2)  res_buf[local_idx] = Operation::red_op(res_buf[local_idx],res_buf[local_idx+1]);
}

template <typename Operation, typename Number>
__global__ void single_vec_reduction(typename Operation::result_type *res, const Number *v, const int N)
{
  __shared__ typename Operation::shmem_type res_buf[SVR_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*SVR_CHUNK_SIZE);
  const int local_idx = threadIdx.x;
  if(global_idx < N)
    res_buf[local_idx] = Operation::elemwise_op(v[global_idx]);
  else
    res_buf[local_idx] = Operation::ident_value();

  for(int c = 1; c < SVR_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*SVR_BKSIZE;
    if(idx < N)
      res_buf[local_idx] = Operation::red_op(res_buf[local_idx],Operation::elemwise_op(v[idx]));
  }

  __syncthreads();

  for(int s = DOT_BKSIZE/2; s>32; s=s>>1) {
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

// all_zero

template <typename Number>
bool GpuVector<Number>::all_zero () const {
  unsigned int *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(unsigned int)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 1, sizeof(unsigned int )));

  const int nblocks = 1 + (_size-1) / (SVR_CHUNK_SIZE*SVR_BKSIZE);
  single_vec_reduction<AllZero<Number> > <<<dim3(nblocks,1),dim3(SVR_BKSIZE,1)>>>(res_d,vec_dev,_size);

  unsigned int res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(unsigned int),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));
  return !(res==0);
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



// scaled addition of vectors
template <typename Number>
void GpuVector<Number>::sadd (const Number a,
                              const Number b,
                              const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_sadd<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,a,b,_size);
}

template <typename Number>
__global__ void add_and_dot_kernel(Number *res, Number *v1, const Number *v2,
                                   const Number *v3, const Number a, const int N)
{
  __shared__ Number res_buf[DOT_BKSIZE];

  const int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*DOT_CHUNK_SIZE);
  const int local_idx = threadIdx.x;
  if(global_idx < N) {

    v1[global_idx]+= a*v2[global_idx];
    res_buf[local_idx] = (v1[global_idx])*v3[global_idx];
  }
  else
    res_buf[local_idx] = 0;

  for(int c = 1; c < DOT_CHUNK_SIZE; ++c) {
    const int idx=global_idx+c*DOT_BKSIZE;
    if(idx < N) {
      v1[idx] += a*v2[idx];
      res_buf[local_idx] += (v1[idx])*v3[idx];
    }
  }

  __syncthreads();

  for(int s = DOT_BKSIZE/2; s>32; s=s>>1) {

    if(local_idx < s)
      res_buf[local_idx] += res_buf[local_idx+s];
    __syncthreads();
  }

  if(local_idx < 32) {
    dotWithinWarp(res_buf,local_idx);
  }

  if(local_idx == 0)
    atomicAdd(res,res_buf[0]);
}

// Combined scaled addition of vector x into
// the current object and subsequent inner
// product of the current object with v
//
// (this + a*x).dot(v)
template <typename Number>
Number GpuVector<Number>::add_and_dot (const Number  a,
                                       const GpuVector<Number> &x,
                                       const GpuVector<Number> &v) {

  Number *res_d;
  CUDA_CHECK_SUCCESS(cudaMalloc(&res_d,sizeof(Number)));
  CUDA_CHECK_SUCCESS(cudaMemset(res_d, 0, sizeof(Number)));

  const int nblocks = 1 + (_size-1) / (DOT_CHUNK_SIZE*DOT_BKSIZE);
  add_and_dot_kernel<Number> <<<dim3(nblocks,1),dim3(DOT_BKSIZE,1)>>>(res_d,vec_dev,x.vec_dev,v.vec_dev,a,_size);

  Number res;
  CUDA_CHECK_SUCCESS(cudaMemcpy(&res,res_d,sizeof(Number),
                                cudaMemcpyDeviceToHost));

  CUDA_CHECK_SUCCESS(cudaFree(res_d));

  return res;
}


// element-wise multiplication of vectors
template <typename Number>
void GpuVector<Number>::scale (const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_bin_op<Number,Binop_Multiplication> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,_size);
}

// element-wise division of vectors
template <typename Number>
GpuVector<Number>& GpuVector<Number>::operator/= (const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_bin_op<Number,Binop_Division> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,_size);
  return *this;
}



// scaled assignment of a vector
template <typename Number>
void GpuVector<Number>::equ (const Number a,
                             const GpuVector<Number> &x) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_equ<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,x.vec_dev,a,_size);
}


// scale the elements of the vector
// by a fixed value
template <typename Number>
GpuVector<Number> & GpuVector<Number>::operator *= (const Number a) {
  const int nblocks = 1 + (_size-1) / (CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
  vec_scale<Number> <<<nblocks,BKSIZE_ELEMWISE_OP>>>(vec_dev,a,_size);
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

  return *this;
}


//=============================================================================
// instantiate templates
//=============================================================================

template class GpuVector<double>;

template class GpuVector<float>;
