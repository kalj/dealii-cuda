/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)cuda_utils.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cstdio>
#include <deal.II/base/exceptions.h>

#ifndef NCUDA_ERROR_CHECK
#define CUDA_CHECK_SUCCESS(errorCall) do {                              \
        cudaError_t return_status=errorCall;                            \
        if(return_status != cudaSuccess) {                              \
            char buf[200];                                              \
            snprintf(buf,sizeof(buf),"Error in %s (%d): %s\n",__FILE__, \
                     __LINE__,cudaGetErrorString(return_status));       \
            AssertThrow(return_status==cudaSuccess, dealii::ExcMessage(buf)); \
        }                                                               \
    } while(0)

#define CUDA_CHECK_LAST CUDA_CHECK_SUCCESS(cudaGetLastError())

#define cudaAssertNoError() cudaThreadSynchronize() ; CUDA_CHECK_SUCCESS(cudaGetLastError())

#else

#define CUDA_CHECK_SUCCESS(errorCall) errorCall
#define CUDA_CHECK_LAST
#define cudaAssertNoError()

#endif


#endif /* _CUDA_UTILS_H */
