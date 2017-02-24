/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)multi_gpu_vec.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 *
 */

#ifndef _MULTI_GPU_VEC_H
#define _MULTI_GPU_VEC_H

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_operations_internal.h>
#include <deal.II/base/subscriptor.h>
#include <cstddef>

#include "gpu_partitioner.h"

namespace dealii {

  template <typename Number>
  class MulitGpuVector : public Subscriptor {
  public:
    typedef types::global_dof_index size_type;
    typedef Number value_type;

    class DevRef {
    private:
      Number *ptr;
      int owning_device;
    public:
      DevRef(Number *p, int owner)
        :ptr(p), owning_device(owner) {}
      DevRef& operator=(const Number value);
    };
  private:

    // contains locally owned values, followed by ghost values
    std::vector<Number *> vec;

    // buffers to receive and send data that is ghosted on other devices
    mutable std::vector<Number *> import_data;

    // the indices of which of my dofs are ghosted on others
    std::vector<unsigned int* > import_indices;

    // size of owned section
    std::vector<unsigned int> local_size;

    // global size
    unsigned int global_size;

    // the underlying paritioner
    std::shared_ptr<const GpuPartitioner> partitioner;

    mutable bool vector_is_ghosted;

    bool vector_is_compressed;
  public:
    // constructors et al.
    MulitGpuVector()
      : global_size(0),
        vector_is_ghosted(false),
        vector_is_compressed(true) {}

    MulitGpuVector(const std::shared_ptr<const GpuPartitioner> &partitioner_in);

    // copy constructor
    MulitGpuVector(const MulitGpuVector<Number>& old);

    // copy constructor from vector based on other number type
    template <typename OtherNumber>
    MulitGpuVector(const MulitGpuVector<OtherNumber>& old);

    // same for assignment
    MulitGpuVector<Number>& operator=(const MulitGpuVector<Number>& old);
    MulitGpuVector<Number>& operator=(const Vector<Number>& old_cpu);
    MulitGpuVector<Number>& operator=(const std::vector<Number>& old_cpu);
    template <typename OtherNumber>
    MulitGpuVector<Number>& operator=(const MulitGpuVector<OtherNumber>& old);

    template <typename OtherNumber> friend class MulitGpuVector;

    ~MulitGpuVector();

    unsigned int size() const { return global_size;}

    // assigns this object in-place to a CPU object
    void copyToHost(Vector<Number>& dst) const;

    // for assignment
    Vector<Number> toVector() const {
      Vector<Number> v(global_size);
      copyToHost(v);
      return v;
    }

    // Access internal data buffer
    // Number *getData() { return vec_dev; }
    // const Number *getDataRO() const { return vec_dev; }

    // initialize with single value
    MulitGpuVector& operator=(const Number n);

    // necessary for deal.ii but shouldn't be used!
    DevRef operator()(const size_t i);

    // necessary for deal.ii but shouldn't be used!
    Number operator()(const size_t i) const;

    // initialize with a partitioner
    void reinit (const std::shared_ptr<const GpuPartitioner> &partitioner_in);

    // resize to have the same structure
    // as the one provided and/or
    // clear vector. note
    // that the second argument must have
    // a default value equal to false
    void reinit (const MulitGpuVector<Number>&,
                 bool leave_elements_uninitialized = false);

    // scalar product
    Number operator * (const MulitGpuVector<Number> &v) const;
    // addition of vectors
    void add (const MulitGpuVector<Number> &V) { sadd(1,1,V); }
    // scaled addition of vectors (this = this + a*V)
    void add (const Number a,
              const MulitGpuVector<Number> &V) { sadd(1,a,V); }
    // scaled addition of vectors (this = s*this + V)
    void sadd (const Number s,
               const MulitGpuVector<Number> &V) { sadd(s,1,V); }
    // scaled addition of vectors (this = s*this + a*V)
    void sadd (const Number s,
               const Number a,
               const MulitGpuVector<Number> &V);


    // addition of vectors
    MulitGpuVector<Number>& operator+=(const MulitGpuVector<Number> &x) { sadd(1,1,x); return (*this); }

    // subtraction of vectors
    MulitGpuVector<Number>& operator-=(const MulitGpuVector<Number> &x) { sadd(1,-1,x); return (*this); }

    // Combined scaled addition of vector x into
    // the current object and subsequent inner
    // product of the current object with v
    Number add_and_dot (const Number  a,
                        const MulitGpuVector<Number> &x,
                        const MulitGpuVector<Number> &v);

    // element-wise multiplication
    void scale(const MulitGpuVector<Number> &v);

    // element-wise division
    MulitGpuVector<Number>& operator/=(const MulitGpuVector<Number> &x);

    // element-wise inversion
    MulitGpuVector<Number>& invert();

    // scaled assignment of a vector
    void equ (const Number a,
              const MulitGpuVector<Number> &x);
    // scale the elements of the vector
    // by a fixed value
    MulitGpuVector<Number> & operator *= (const Number a);

    // return the l2 norm of the vector
    Number l2_norm () const;

    unsigned int memory_consumption() const
    {
      // return _size*sizeof(Number);
      return 0;
    }

    // are all entries zero?
    bool all_zero() const;

    void print(std::ostream &out, const unsigned int precision = 3,
               const bool scientific = true, const bool across = true) const {
      toVector().print(out,precision,scientific,across);
    }

    void swap(MulitGpuVector<Number> &other);
 // {
 //      Number * tmp_vec = vec_dev;
 //      unsigned int tmp_size = _size;
 //      vec_dev = other.vec_dev;
 //      _size = other._size;

 //      other.vec_dev = tmp_vec;
 //      other._size = tmp_size;
 //    }

    IndexSet locally_owned_elements() const { return complete_index_set(size()); }

    void compress(VectorOperation::values   operation = VectorOperation::unknown);

    void update_ghost_values() const;
  };
}

#endif /* _MULTI_GPU_VEC_H */
