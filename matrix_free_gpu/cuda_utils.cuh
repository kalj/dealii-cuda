/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)cuda_utils.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <cstdio>

#ifndef NCUDA_ERROR_CHECK
#define CUDA_CHECK_SUCCESS(errorCall) do {                        \
        cudaError_t error_variable=errorCall;                     \
        if(cudaSuccess != error_variable) {                       \
            fprintf(stderr,"Error in %s (%d): %s\n",__FILE__,     \
                    __LINE__,cudaGetErrorString(error_variable)); \
            exit(1);                                              \
        }                                                         \
    } while(0)

#define CUDA_CHECK_LAST CUDA_CHECK_SUCCESS(cudaGetLastError())

#else

#define CUDA_CHECK_SUCCESS(errorCall) errorCall
#define CUDA_CHECK_LAST

#endif


#endif /* _CUDA_UTILS_H */
