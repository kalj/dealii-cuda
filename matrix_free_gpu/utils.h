/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)utils.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _UTILS_H
#define _UTILS_H

template<int a, int n>
struct ipow {
    static int const val = a*ipow<a,n-1>::val;
};


template<int a>
struct ipow<a,0> {
    static int const val = 1;
};

__host__ __device__ constexpr int ipowf(int a, int n) {
  return n==0 ? 1 : a*ipowf(a,n-1);
}


#endif /* _UTILS_H */
