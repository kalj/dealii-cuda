#ifndef __deal2__gpu_array_h
#define __deal2__gpu_array_h

#include "maybecuda.h"

/*
 * This is essentially a GPU version of Point, i.e. a constant-length vector / tuple.
 *
 */

template <int size, typename Number>
class GpuArray {
public:
  Number arr[size];

  MAYBECUDA_HOSTDEV const Number &operator[](const unsigned int i) const { return arr[i]; }
  MAYBECUDA_HOSTDEV Number &operator[](const unsigned int i) { return arr[i]; }
  MAYBECUDA_HOSTDEV Number norm_square() const {
    Number apa =0;
    for(int i = 0; i < size; ++i) apa+=arr[i]*arr[i];
    return apa;
  }

  MAYBECUDA_HOSTDEV GpuArray<size,Number> operator-() const;
};

template <int size, typename Number>
MAYBECUDA_HOSTDEV GpuArray<size,Number> GpuArray<size,Number>::operator-() const
{
  GpuArray<size,Number> a;
  for(int i = 0; i < size; ++i) a.arr[i] = -arr[i];
  return a;
}

template <int size, typename Number>
MAYBECUDA_HOSTDEV GpuArray<size,Number> operator*(const GpuArray<size,Number>&v, Number a)
{
  GpuArray<size,Number> res;
  for(int i = 0; i < size; ++i) {
    res.arr[i] = a*v.arr[i];
  }
  return res;
}

template <int size, typename Number>
inline MAYBECUDA_HOSTDEV GpuArray<size,Number> operator*(Number a, const GpuArray<size,Number>&v)
{
  return v*a;
}

// template <int size, typename Number>
// MAYBECUDA_HOSTDEV GpuArray<size,Number> operator*(const GpuArray<size,Number>&v, const GpuArray<size,Number>&v2)
// {
//     GpuArray<size,Number> res;
//     for(int i = 0; i < size; ++i) {
//         res.arr[i] = v.arr[i]*v2.arr[i];
//     }
//     return res;
// }


#endif
