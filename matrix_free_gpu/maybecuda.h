/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)maybecuda.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _MAYBECUDA_H
#define _MAYBECUDA_H

#ifdef __CUDACC__

// using nvcc

#define MAYBECUDA_GUARD
#define MAYBECUDA_HOSTDEV __host__ __device__
#define MAYBECUDA_HOST __host__

#else

// not using nvcc

#define MAYBECUDA_HOSTDEV
#define MAYBECUDA_HOST

#endif

#endif /* _MAYBECUDA_H */
