/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)tensor_ops.cuh
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 *
 */

#ifndef _TENSOR_OPS_H
#define _TENSOR_OPS_H

#include "defs.h"
#include "utils.h"


//=============================================================================
// This implements tensor contractions in 1, 2, and 3 dimensions. Also included
// are aggregates for integrating a scalar function or its gradient, or to
// interpolate values at quadrature points.
//=============================================================================

template <int dim, int n, typename Number>
struct TensorOpsShmem {

  template <int redidx, bool add, bool inplace>
  static inline __device__ void contraction(Number *dst, const Number *src, const Number *shape_buf)
  {
    assert(redidx >= 0 && redidx < dim);

    if(dim == 1)
    {
      const unsigned int q = (threadIdx.x%n); // new index

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index
          t += shape_buf[k] * (inplace ? dst[k] : src[k]);
      }

      if(inplace) __syncthreads();

      if(add)
        dst[q] += t;
      else
        dst[q] = t;

    }
    else if(dim==2)
    {
      const unsigned int q = threadIdx.x; // new index
      const unsigned int i = (threadIdx.y%n); // unchanged index

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index

        const unsigned int srcidx =
          (redidx==0) ? (k + n*i)
          : (i + n*k);

          t += shape_buf[k] * (inplace ? dst[srcidx] : src[srcidx]);
      }

      if(inplace) __syncthreads();

      const unsigned int dstidx =
        (redidx==0) ? (q + n*i)
        : (i + n*q);

      if(add)
        dst[dstidx] += t;
      else
        dst[dstidx] = t;
    }
    else if(dim==3)
    {
      const unsigned int q = threadIdx.x; // new index
      const unsigned int i = threadIdx.y; // two unchanged
      const unsigned int j = threadIdx.z%n; // indices

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index

        const unsigned int srcidx =
          (redidx==0) ? (k + n*(i + n*j))
          : (redidx==1) ? (i + n*(k + n*j))
          : (i + n*(j + n*k));

          t += shape_buf[k] * (inplace ? dst[srcidx] : src[srcidx]);
      }

      if(inplace) __syncthreads();

      const unsigned int dstidx =
        (redidx==0) ? (q + n*(i + n*j))
        : (redidx==1) ? (i + n*(q + n*j))
        : (i + n*(j + n*q));

      if(add)
        dst[dstidx] += t;
      else
        dst[dstidx] = t;
    }
  }


  static inline __device__ void fun_at_quad_pts(Number *u)
  {
    // Stage column of shape_values:
    Number shape_values_buf[n];

#pragma unroll
    for(int i = 0; i < n; i++) {
      shape_values_buf[i] = shape_values[threadIdx.x + n*i];
    }


    if(dim==1) {
      contraction<0,false,true> (u,u,shape_values_buf);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (u,u,shape_values_buf);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along z / k / q - direction
      contraction<2,false,true> (u,u,shape_values_buf);
    }
  }


  static inline __device__ void quad_int_fun(Number *u)
  {
    // Stage row of shape_values:
    Number shape_values_buf[n];

#pragma unroll
    for(int i = 0; i < n; i++) {
      shape_values_buf[i] = shape_values[threadIdx.x*n+i];
    }

    if(dim==1) {
      contraction<0,false,true> (u,u,shape_values_buf);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (u,u,shape_values_buf);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (u,u,shape_values_buf);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,false,true> (u,u,shape_values_buf);
    }
  }

  static inline __device__ void grad_at_quad_pts(Number *duq[dim], const Number * const u)
  {
    // Stage column of shape_values:
    Number shape_values_buf[n];
    Number shape_gradient_buf[n];

#pragma unroll
    for(int i = 0; i < n; i++) {
      shape_values_buf[i] = shape_values[threadIdx.x + n*i];
      shape_gradient_buf[i] = shape_gradient[threadIdx.x + n*i];
    }


    if(dim==1) {
      contraction<0,false,false> (duq[0],u,shape_gradient_buf);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,false> (duq[0],u,shape_gradient_buf);
      contraction<0,false,false> (duq[1],u,shape_values_buf);

      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (duq[0],duq[0],shape_values_buf);
      contraction<1,false,true>  (duq[1],duq[1],shape_gradient_buf);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,false> (duq[0],u,shape_gradient_buf);
      contraction<0,false,false> (duq[1],u,shape_values_buf);
      contraction<0,false,false> (duq[2],u,shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (duq[0],duq[0],shape_values_buf);
      contraction<1,false,true>  (duq[1],duq[1],shape_gradient_buf);
      contraction<1,false,true> (duq[2],duq[2],shape_values_buf);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,false,true> (duq[0],duq[0],shape_values_buf);
      contraction<2,false,true> (duq[1],duq[1],shape_values_buf);
      contraction<2,false,true>  (duq[2],duq[2],shape_gradient_buf);
    }
  }

  template <bool add>
  static inline __device__ void quad_int_grad(Number *u, Number * duq[dim])
  {
    // Stage row of shape_values:
    Number shape_values_buf[n];
    Number shape_gradient_buf[n];

#pragma unroll
    for(int i = 0; i < n; i++) {
      shape_values_buf[i] = shape_values[threadIdx.x*n + i];
      shape_gradient_buf[i] = shape_gradient[threadIdx.x*n + i];
    }


    if(dim==1) {
      contraction<0,add,false> (u,duq[0],shape_gradient_buf);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,true> (duq[0],duq[0],shape_gradient_buf);
      contraction<0,false,true> (duq[1],duq[1],shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,add,false> (u,duq[0],shape_values_buf);
      __syncthreads();
      contraction<1,true,false> (u,duq[1],shape_gradient_buf);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,true>  (duq[0],duq[0],shape_gradient_buf);
      contraction<0,false,true> (duq[1],duq[1],shape_values_buf);
      contraction<0,false,true> (duq[2],duq[2],shape_values_buf);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,true> (duq[0],duq[0],shape_values_buf);
      contraction<1,false,true>  (duq[1],duq[1],shape_gradient_buf);
      contraction<1,false,true> (duq[2],duq[2],shape_values_buf);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,add,false> (u,duq[0],shape_values_buf);
      __syncthreads();
      contraction<2,true,false> (u,duq[1],shape_values_buf);
      __syncthreads();
      contraction<2,true,false> (u,duq[2],shape_gradient_buf);

    }
  }
};



#endif /* _TENSOR_OPS_H */
