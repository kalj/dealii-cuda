
#include "cuda_memory_utils.h"
#include "cuda_utils.cuh"

#define CUDA_MEMORY_UTILS_ARRAY_COPY_BKSIZE 256

namespace dealii
{
  namespace internal
  {
    namespace kernels
    {
      template <typename DstNumber, typename SrcNumber>
      __global__ void copy_dev_array(DstNumber *dst,
                                     const SrcNumber *src,
                                     const unsigned len);
    }


    template <typename DstNumber, typename SrcNumber>
    void copy_dev_array(DstNumber *dst,
                        const SrcNumber *src,
                        const unsigned len)
    {
      const unsigned int nblocks = 1 + (len-1) / CUDA_MEMORY_UTILS_ARRAY_COPY_BKSIZE;

      if(len>0) {
        kernels::copy_dev_array<<<nblocks,CUDA_MEMORY_UTILS_ARRAY_COPY_BKSIZE>>>(dst,src,len);
        CUDA_CHECK_LAST;
      }
    }


    namespace kernels
    {
      template <typename DstNumber, typename SrcNumber>
      __global__ void copy_dev_array(DstNumber *dst,
                                     const SrcNumber *src,
                                     const unsigned len)
      {
        const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
        if(idx < len) {
          dst[idx] = src[idx];
        }
      }
    }
  }
}

namespace dealii
{
  namespace internal
  {
      template void copy_dev_array<double,float>(double *dst,
                                                 const float *src,
                                                 const unsigned len);

      template void copy_dev_array<float,double>(float *dst,
                                                 const double *src,
                                                 const unsigned len);
  }
}