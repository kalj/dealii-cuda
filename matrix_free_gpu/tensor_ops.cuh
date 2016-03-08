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

template <int dim, int n, typename Number>
struct TensorOpsShmem {

  template <int redidx, bool phi_tr, bool add, bool with_gradients, bool inplace>
  static inline __device__ void contraction(Number *dst, const Number *src)
  {
    assert(redidx >= 0 && redidx < dim);

    if(dim == 1)
    {
      const unsigned int q = (threadIdx.x%n); // new index

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index

        const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);

        if(with_gradients)
          t += shape_gradient[shapeidx] * (inplace ? dst[k] : src[k]);
        else
          t += shape_values[shapeidx] * (inplace ? dst[k] : src[k]);
      }

      if(inplace) __syncthreads();

      if(add)
        dst[q] += t;
      else
        dst[q] = t;

    }
    else if(dim==2)
    {
      const unsigned int i = (threadIdx.x%n); // unchanged index
      const unsigned int q = threadIdx.y; // new index

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index

        const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);
        const unsigned int srcidx =
          (redidx==0) ? (k + n*i)
          : (i + n*k);

        if(with_gradients)
          t += shape_gradient[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
        else
          t += shape_values[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
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
      const unsigned int i = (threadIdx.x%n); // two unchanged
      const unsigned int j = threadIdx.y; // indices
      const unsigned int q = threadIdx.z; // new index

      Number t = 0;
      for(int k = 0; k < n; ++k) { // contracted index

        const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);
        const unsigned int srcidx =
          (redidx==0) ? (k + n*(i + n*j))
          : (redidx==1) ? (i + n*(k + n*j))
          : (i + n*(j + n*k));

        if(with_gradients)
          t += shape_gradient[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
        else
          t += shape_values[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
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
    if(dim==1) {
      contraction<0,true,false,false,true> (u,u);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,true,false,false,true> (u,u);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,true,false,false,true> (u,u);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,true,false,false,true> (u,u);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,true,false,false,true> (u,u);
      __syncthreads();

      // reduce along z / k / q - direction
      contraction<2,true,false,false,true> (u,u);
    }
  }


  static inline __device__ void quad_int_fun(Number *u)
  {
    if(dim==1) {
      contraction<0,false,false,false,true> (u,u);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,false,false,true> (u,u);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,false,false,true> (u,u);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,false,false,true> (u,u);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,false,false,true> (u,u);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,false,false,false,true> (u,u);
    }
  }

  static inline __device__ void grad_at_quad_pts(Number *duq[dim], const Number * const u)
  {
    if(dim==1) {
      contraction<0,true,false,true,false> (duq[0],u);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,true,false,true,false> (duq[0],u);
      contraction<0,true,false,false,false> (duq[1],u);

      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,true,false,false,true> (duq[0],duq[0]);
      contraction<1,true,false,true,true> (duq[1],duq[1]);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,true,false,true,false> (duq[0],u);
      contraction<0,true,false,false,false> (duq[1],u);
      contraction<0,true,false,false,false> (duq[2],u);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,true,false,false,true> (duq[0],duq[0]);
      // contraction<1,true,false,false,false> (duq[2],duq[1]);
      // __syncthreads();
      contraction<1,true,false,true,true>  (duq[1],duq[1]);
      contraction<1,true,false,false,true> (duq[2],duq[2]);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,true,false,false,true> (duq[0],duq[0]);
      contraction<2,true,false,false,true> (duq[1],duq[1]);
      contraction<2,true,false,true,true>  (duq[2],duq[2]);
    }
  }

  template <bool add>
  static inline __device__ void quad_int_grad(Number *u, Number * duq[dim])
  {
    if(dim==1) {
      contraction<0,false,add,true,false> (u,duq[0]);
    }
    else if(dim==2) {

      // reduce along x / i / q - direction
      contraction<0,false,false,true,true> (duq[0],duq[0]);
      contraction<0,false,false,false,true> (duq[1],duq[1]);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,add,false,false> (u,duq[0]);
      __syncthreads();
      contraction<1,false,true,true,false> (u,duq[1]);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,false,true,true>  (duq[0],duq[0]);
      contraction<0,false,false,false,true> (duq[1],duq[1]);
      contraction<0,false,false,false,true> (duq[2],duq[2]);
      __syncthreads();

      // reduce along y / j / r - direction
      contraction<1,false,false,false,true> (duq[0],duq[0]);
      // contraction<1,false,false,false,false> (duq[2],duq[1]);
      // __syncthreads();
      contraction<1,false,false,true,true>  (duq[1],duq[1]);
      contraction<1,false,false,false,true> (duq[2],duq[2]);
      __syncthreads();

      // reduce along z / k / s - direction
      contraction<2,false,add,false,false> (u,duq[0]);
      __syncthreads();
      contraction<2,false,true,false,false> (u,duq[1]);
      __syncthreads();
      contraction<2,false,true,true,false> (u,duq[2]);

    }
  }
};

template <int dim, int n, typename Number>
struct TensorOpsPmem {

  template <int redidx, bool phi_tr, bool add, bool with_gradients, bool inplace>
  static inline __device__ void contraction(Number *dst, const Number *src)
  {
    assert(redidx >= 0 && redidx < dim);

    if(dim==1)
    {
      for(int q = 0; q < n; ++q) { // new index

        Number t = 0;
        for(int k = 0; k < n; ++k) { // contracted index

          const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);

          if(with_gradients)
            t += shape_gradient[shapeidx] * (inplace ? dst[k] : src[k]);
          else
            t += shape_values[shapeidx] * (inplace ? dst[k] : src[k]);
        }

        if(add)
          dst[q] += t;
        else
          dst[q] = t;
      }
    }
    else if(dim==2)
    {
      for(int i = 0; i < n; ++i) { // one unchanged index
        for(int q = 0; q < n; ++q) { // new index

          Number t = 0;
          for(int k = 0; k < n; ++k) { // contracted index

            const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);
            const unsigned int srcidx =
              (redidx==0) ? (k + n*i)
              : (i + n*k);

            if(with_gradients)
              t += shape_gradient[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
            else
              t += shape_values[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
          }

          const unsigned int dstidx =
            (redidx==0) ? (q + n*i)
            : (i + n*q);

          if(add)
            dst[dstidx] += t;
          else
            dst[dstidx] = t;

        }
      }
    }
    else if(dim==3)
    {
      for(int i = 0; i < n; ++i) { // two unchanged
        for(int j = 0; j < n; ++j) { // indices
          for(int q = 0; q < n; ++q) { // new index

            Number t = 0;
            for(int k = 0; k < n; ++k) { // contracted index

              const unsigned int shapeidx = phi_tr ? (q+k*n) : (k+q*n);
              const unsigned int srcidx =
                (redidx==0) ? (k + n*(i + n*j))
                : (redidx==1) ? (i + n*(k + n*j))
                : (i + n*(j + n*k));

              if(with_gradients)
                t += shape_gradient[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
              else
                t += shape_values[shapeidx] * (inplace ? dst[srcidx] : src[srcidx]);
            }

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
      }
    }
  }


  static inline __device__ void fun_at_quad_pts(Number *uq,const Number *u)
  {
    if(dim==1) {
      contraction<0,true,false,false,false> (uq,u);
    }
    else if(dim==2) {

      Number tmp[n*n];
      // reduce along x / i / q - direction
      contraction<0,true,false,false,false> (tmp,u);

      // reduce along y / j / r - direction
      contraction<1,true,false,false,false> (uq,tmp);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,true,false,false,false> (uq,u);
      Number tmp[n*n*n];

      // reduce along y / j / r - direction
      contraction<1,true,false,false,false> (tmp,uq);

      // reduce along z / k / q - direction
      contraction<2,true,false,false,false> (uq,tmp);
    }
  }

  static inline __device__ void quad_int_fun(Number *u, const Number *uq)
  {
    if(dim==1) {
      contraction<0,false,false,false,false> (u,uq);
    }
    else if(dim==2) {

      Number tmp[n*n];
      // reduce along x / i / q - direction
      contraction<0,false,false,false,false> (tmp,uq);

      // reduce along y / j / r - direction
      contraction<1,false,false,false,false> (u,tmp);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,false,false,false,false> (u,uq);
      Number tmp[n*n*n];

      // reduce along y / j / r - direction
      contraction<1,false,false,false,false> (tmp,u);

      // reduce along z / k / s - direction
      contraction<2,false,false,false,false> (u,tmp);
    }
  }

  static inline __device__ void grad_at_quad_pts(Number duq[dim][ipowf(n,dim)], const Number *u)
  {
    if(dim==1) {
      contraction<0,true,false,true,false> (duq[0],u);
    }
    else if(dim==2) {

      Number tmp[n*n];

      // reduce along x / i / q - direction
      contraction<0,true,false,true,false> (tmp,u);
      // reduce along y / j / r - direction
      contraction<1,true,false,false,false> (duq[0],tmp);

      // reduce along x / i / q - direction
      contraction<0,true,false,false,false> (tmp,u);
      // reduce along y / j / r - direction
      contraction<1,true,false,true,false> (duq[1],tmp);
    }
    else if(dim==3) {

      // reduce along x / i / q - direction
      contraction<0,true,false,true,false> (duq[0],u);
      // reduce along x / i / q - direction
      contraction<0,true,false,false,false> (duq[1],u);

      Number tmp[n*n*n];

      // reduce along y / j / r - direction
      contraction<1,true,false,false,false> (tmp,duq[0]);
      // reduce along z / k / s - direction
      contraction<2,true,false,false,false> (duq[0],tmp);

      // reduce along y / j / r - direction
      contraction<1,true,false,false,false> (tmp,duq[1]);
      // reduce along z / k / s - direction
      contraction<2,true,false,true,false> (duq[2],tmp);

      // reduce along y / j / r - direction
      contraction<1,true,false,true,false> (tmp,duq[1]);
      // reduce along z / k / s - direction
      contraction<2,true,false,false,false> (duq[1],tmp);
    }
  }

  template <bool add>
  static inline __device__ void quad_int_grad(Number *u, const Number duq[dim][ipowf(n,dim)])
  {
    if(dim==1) {
      contraction<0,false,add,true,false> (u,duq[0]);
    }
    else if(dim==2) {

      Number tmp[n*n];

      // reduce along x / i / q - direction
      contraction<0,false,false,true,false> (tmp,duq[0]);
      // reduce along y / j / r - direction
      contraction<1,false,add,false,false> (u,tmp);

      // reduce along x / i / q - direction
      contraction<0,false,false,false,false> (tmp,duq[1]);
      // reduce along y / j / r - direction
      contraction<1,false,true,true,false> (u,tmp);
    }
    else if(dim==3) {

      Number tmp[n*n*n];
      Number tmp2[n*n*n];

      // reduce along x / i / q - direction
      contraction<0,false,false,true,false> (tmp,duq[0]);
      // reduce along y / j / r - direction
      contraction<1,false,false,false,false> (tmp2,tmp);
      // reduce along z / k / s - direction
      contraction<2,false,add,false,false> (u,tmp2);

      // reduce along x / i / q - direction
      contraction<0,false,false,false,false> (tmp,duq[1]);
      // reduce along y / j / r - direction
      contraction<1,false,false,true,false> (tmp2,tmp);
      // reduce along z / k / s - direction
      contraction<2,false,true,false,false> (u,tmp2);

      // reduce along x / i / q - direction
      contraction<0,false,false,false,false> (tmp,duq[2]);
      // reduce along y / j / r - direction
      contraction<1,false,false,false,false> (tmp2,tmp);
      // reduce along z / k / s - direction
      contraction<2,false,true,true,false> (u,tmp2);
    }
  }
};




#endif /* _TENSOR_OPS_H */
