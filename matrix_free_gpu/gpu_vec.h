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
using namespace dealii;

template <typename Number>
class GpuVector : public Subscriptor {
public:
  typedef types::global_dof_index size_type;
private:
  Number *vec_dev;
  int _size;
public:
  // constructors et al.
  GpuVector()
    : vec_dev(NULL), _size(0) {}

  // initialize with size
  GpuVector(unsigned int s);

  // copy constructor
  GpuVector(const GpuVector<Number>& old);

  // copy constructor from CPU object
  GpuVector(const Vector<Number>& old_cpu);

  // same for assignment
  GpuVector<Number>& operator=(const GpuVector<Number>& old);
  GpuVector<Number>& operator=(const Vector<Number>& old_cpu);

  ~GpuVector();

  int size() const { return _size;}

  // assigns this object in-place to a CPU object
  void copyToHost(Vector<Number>& dst) const;

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

  // necessary for deal.ii but not allowed here!
  Number& operator()(const size_t i) {
    ExcNotImplemented();
    return vec_dev[0];
  }

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
  void add (const GpuVector<Number> &x) { sadd(1,1,x); }
  // scaled addition of vectors (this = this + a*x)
  void add (const Number a,
            const GpuVector<Number> &x) { sadd(1,a,x); }
  // scaled addition of vectors (this = a*this + b*x)
  void sadd (const Number a,
             const Number b,
             const GpuVector<Number> &x);
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

#endif /* _GPUVEC_H */
