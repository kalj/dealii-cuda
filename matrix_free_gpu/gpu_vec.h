/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)gpu_vec.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 *
 */

#ifndef _GPUVEC_H
#define _GPUVEC_H

#include <deal.II/lac/vector.h>
#include <deal.II/base/subscriptor.h>
#include <cstddef>

#include "gpu_list.h"

using namespace dealii;

// forward declarations
template <typename Number> class VecSlice;
template <typename Number> class ConstVecSlice;
template <typename Number> class VecSlicePlusVecSlice;
template <typename Number> class VecSlicePlusVec;
template <typename Number> class VecPlusVec;

template <typename Number>
class GpuVector : public Subscriptor {
public:
  typedef types::global_dof_index size_type;
  typedef Number value_type;

  class DevRef {
  private:
    Number *ptr;
  public:
    DevRef(Number *p)
      :ptr(p) {}
    DevRef& operator=(const Number value);
  };
private:
  Number *vec_dev;
  unsigned int _size;
public:
  // constructors et al.
  GpuVector()
    : vec_dev(NULL), _size(0) {}

  // initialize with size. This sets all elements to zero
  GpuVector(unsigned int s);

  // copy constructor
  GpuVector(const GpuVector<Number>& old);

  // copy constructor from CPU object
  GpuVector(const Vector<Number>& old_cpu);

  GpuVector(const std::vector<Number>& old_cpu);

  // same for assignment
  GpuVector<Number>& operator=(const GpuVector<Number>& old);
  GpuVector<Number>& operator=(const Vector<Number>& old_cpu);
  GpuVector<Number>& operator=(const std::vector<Number>& old_cpu);

  ~GpuVector();

  unsigned int size() const { return _size;}

  // assigns this object in-place to a CPU object
  void copyToHost(Vector<Number>& dst) const;

  void fromHost(const Number *buf, unsigned int n);

  // for assignment
  Vector<Number> toVector() const {
    Vector<Number> v(_size);
    copyToHost(v);
    return v;
  }

  // Access internal data buffer
  Number *getData() { return vec_dev; }
  const Number *getDataRO() const { return vec_dev; }

  // initialize with single value
  GpuVector& operator=(const Number n);

  // necessary for deal.ii but shouldn't be used!
  DevRef operator()(const size_t i);

  // necessary for deal.ii but shouldn't be used!
  Number operator()(const size_t i) const;

  // resize to have the same structure
  // as the one provided and/or
  // clear vector. note
  // that the second argument must have
  // a default value equal to false
  void reinit (const GpuVector<Number>&,
               bool leave_elements_uninitialized = false);

  // scalar product
  Number operator * (const GpuVector<Number> &v) const;
  // addition of vectors
  void add (const GpuVector<Number> &V) { sadd(1,1,V); }
  // scaled addition of vectors (this = this + a*V)
  void add (const Number a,
            const GpuVector<Number> &V) { sadd(1,a,V); }
  // scaled addition of vectors (this = s*this + V)
  void sadd (const Number s,
             const GpuVector<Number> &V) { sadd(s,1,V); }
  // scaled addition of vectors (this = s*this + a*V)
  void sadd (const Number s,
             const Number a,
             const GpuVector<Number> &V);


  // operations with slices
  VecSlice<Number> operator[](const GpuList<unsigned int> indices);
  const ConstVecSlice<Number> operator[](const GpuList<unsigned int> indices) const;
  GpuVector<Number>& operator=(const ConstVecSlice<Number> &other);
  GpuVector<Number>& operator=(const VecSlicePlusVecSlice<Number> &other);
  GpuVector<Number>& operator=(const VecSlicePlusVec<Number> &other);

  // addition of vectors
  GpuVector<Number>& operator+=(const GpuVector<Number> &x) { sadd(1,1,x); return (*this); }

  // subtraction of vectors
  GpuVector<Number>& operator-=(const GpuVector<Number> &x) { sadd(1,-1,x); return (*this); }

  // Combined scaled addition of vector x into
  // the current object and subsequent inner
  // product of the current object with v
  Number add_and_dot (const Number  a,
                      const GpuVector<Number> &x,
                      const GpuVector<Number> &v);

  // element-wise multiplication
  void scale(const GpuVector<Number> &v);

  // element-wise division
  GpuVector<Number>& operator/=(const GpuVector<Number> &x);

  // scaled assignment of a vector
  void equ (const Number a,
            const GpuVector<Number> &x);
  // scale the elements of the vector
  // by a fixed value
  GpuVector<Number> & operator *= (const Number a);

  // return the l2 norm of the vector
  Number l2_norm () const;

  unsigned int memory_consumption() const {return _size*sizeof(Number); }

  void resize(unsigned int);

  // are all entries zero?
  bool all_zero() const;

  void print(std::ostream &out, const unsigned int precision = 3,
             const bool scientific = true, const bool across = true) const {
    toVector().print(out,precision,scientific,across);
  }

  void swap(GpuVector<Number> &other) {
    Number * tmp_vec = vec_dev;
    unsigned int tmp_size = _size;
    vec_dev = other.vec_dev;
    _size = other._size;

    other.vec_dev = tmp_vec;
    other._size = tmp_size;
  }

  IndexSet locally_owned_elements() const { return complete_index_set(size()); }
  void compress(::VectorOperation::values   operation = ::VectorOperation::unknown) const { }
};


template <typename Number>
struct VecSlice
{
  GpuVector<Number> &vec;
  const GpuList<unsigned int> &idx;

  VecSlice(GpuVector<Number> &v,
           const GpuList<unsigned int> &i);

  VecSlice<Number>& operator=(const ConstVecSlice<Number> &other);
  VecSlice<Number>& operator=(const VecSlicePlusVecSlice<Number> &other);
  VecSlice<Number>& operator=(const VecSlicePlusVec<Number> &other);
  VecSlice<Number>& operator=(const VecPlusVec<Number> &other);
  VecSlice<Number>& operator=(const Number a);
  VecSlice<Number>& operator=(const GpuVector<Number> &other);
};

template <typename Number>
struct ConstVecSlice
{
  const GpuVector<Number> &vec;
  const GpuList<unsigned int> &idx;

  ConstVecSlice(const GpuVector<Number> &v,
                const GpuList<unsigned int> &i);
  // ConstVecSlice(const VecSlice<Number>& other);
};


template <typename Number>
struct VecSlicePlusVecSlice
{
  const GpuVector<Number> &vec1;
  const GpuList<unsigned int> &idx1;
  const GpuVector<Number> &vec2;
  const GpuList<unsigned int> &idx2;

  VecSlicePlusVecSlice(const GpuVector<Number> &v1,
                       const GpuList<unsigned int> &i1,
                       const GpuVector<Number> &v2,
                       const GpuList<unsigned int> &i2);
};

template <typename Number>
struct VecSlicePlusVec
{
  const GpuVector<Number> &vec1;
  const GpuList<unsigned int> &idx1;
  const GpuVector<Number> &vec2;

  VecSlicePlusVec(const GpuVector<Number> &v1,
                  const GpuList<unsigned int> &i1,
                  const GpuVector<Number> &v2);
};

template <typename Number>
struct VecPlusVec
{
  const GpuVector<Number> &vec1;
  const GpuVector<Number> &vec2;

  VecPlusVec(const GpuVector<Number> &v1,
             const GpuVector<Number> &v2);
};


template <typename Number>
VecSlicePlusVecSlice<Number> operator+(const ConstVecSlice<Number> l, const ConstVecSlice<Number> r);

template <typename Number>
VecSlicePlusVec<Number> operator+(const ConstVecSlice<Number> l, const GpuVector<Number>& r);

template <typename Number>
VecPlusVec<Number> operator+(const GpuVector<Number>& l, const GpuVector<Number>& r);

template <typename Number>
void copy_with_indices(GpuVector<Number> &dst, const GpuVector<Number> &src,
                       const GpuList<int> &dst_indices, const GpuList<int> &src_indices);



#endif /* _GPUVEC_H */
