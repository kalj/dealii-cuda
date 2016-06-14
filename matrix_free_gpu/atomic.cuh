/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)atomic.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _ATOMIC_CUH
#define _ATOMIC_CUH

inline __device__ double atomicAddWrapper(double* address, double val)
{
#if  __CUDACC_VER_MAJOR__  >= 8 && ( !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600 )
  // use native instruction
  return atomicAdd(address,val);
#else

  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

inline __device__  float atomicAddWrapper(float* address, float val)
{
  return atomicAdd(address,val);
}


#endif /* _ATOMIC_CUH */
