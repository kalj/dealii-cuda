/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)atomic.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _ATOMIC_CUH
#define _ATOMIC_CUH

inline __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}



#endif /* _ATOMIC_CUH */
