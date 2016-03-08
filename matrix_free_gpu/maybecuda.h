/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)maybecuda.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _MAYBECUDA_H
#define _MAYBECUDA_H

#ifdef __CUDACC__
// #warning Using nvcc
#define MAYBECUDA_HOSTDEV __host__ __device__
#else
// #warning Not using nvcc
#define MAYBECUDA_HOSTDEV
#endif

#endif /* _MAYBECUDA_H */
