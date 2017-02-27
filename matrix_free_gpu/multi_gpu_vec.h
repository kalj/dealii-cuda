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
#include "multi_gpu_list.h"


namespace dealii
{

  template <typename Number>
  class MultiGpuVector : public Subscriptor {
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
    mutable MultiGpuList<Number> import_data;

    // the indices of which of my dofs are ghosted on others
    MultiGpuList<unsigned int> import_indices;

    // size of owned section
    std::vector<unsigned int> local_sizes;

    // global size
    unsigned int global_size;

    // the underlying paritioner
    std::shared_ptr<const GpuPartitioner> partitioner;

    mutable bool vector_is_ghosted;

    bool vector_is_compressed;
  public:
    // constructors et al.
    MultiGpuVector();

    MultiGpuVector(const std::shared_ptr<const GpuPartitioner> &partitioner_in);

    // copy constructor
    MultiGpuVector(const MultiGpuVector<Number>& old);

    // copy constructor from vector based on other number type
    template <typename OtherNumber>
    MultiGpuVector(const MultiGpuVector<OtherNumber>& old);

    const std::shared_ptr<const GpuPartitioner> &get_partitioner() const
    {
      return partitioner;
    }

    // same for assignment
    MultiGpuVector<Number>& operator=(const MultiGpuVector<Number>& old);
    MultiGpuVector<Number>& operator=(const Vector<Number>& old_cpu);
    MultiGpuVector<Number>& operator=(const std::vector<Number>& old_cpu);

    template <typename OtherNumber>
    MultiGpuVector<Number>& operator=(const MultiGpuVector<OtherNumber>& old);

    template <typename OtherNumber> friend class MultiGpuVector;

    ~MultiGpuVector();

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
    Number *getData(unsigned int part=0) { return vec[part]; }
    const Number *getDataRO(unsigned int part=0) const { return vec[part]; }

    // initialize with single value
    MultiGpuVector& operator=(const Number n);

    // necessary for deal.ii but shouldn't be used!
    DevRef operator()(const size_t i);

    // necessary for deal.ii but shouldn't be used!
    Number operator()(const size_t i) const;

    // only there for compatibility, only s==0 is valid which is equivalend to
    // a clear
    void reinit (unsigned int s);

    // initialize with a partitioner
    void reinit (const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                 bool leave_elements_uninitialized = false);

    // resize to have the same structure
    // as the one provided and/or
    // clear vector. note
    // that the second argument must have
    // a default value equal to false
    void reinit (const MultiGpuVector<Number>&,
                 bool leave_elements_uninitialized = false);


    // scalar product
    Number operator * (const MultiGpuVector<Number> &v) const;
    // addition of vectors
    void add (const MultiGpuVector<Number> &V) { sadd(1,1,V); }
    // scaled addition of vectors (this = this + a*V)
    void add (const Number a,
              const MultiGpuVector<Number> &V) { sadd(1,a,V); }
    // scaled addition of vectors (this = s*this + V)
    void sadd (const Number s,
               const MultiGpuVector<Number> &V) { sadd(s,1,V); }
    // scaled addition of vectors (this = s*this + a*V)
    void sadd (const Number s,
               const Number a,
               const MultiGpuVector<Number> &V);


    // addition of vectors
    MultiGpuVector<Number>& operator+=(const MultiGpuVector<Number> &x) { sadd(1,1,x); return (*this); }

    // subtraction of vectors
    MultiGpuVector<Number>& operator-=(const MultiGpuVector<Number> &x) { sadd(1,-1,x); return (*this); }

    // Combined scaled addition of vector x into
    // the current object and subsequent inner
    // product of the current object with v
    Number add_and_dot (const Number  a,
                        const MultiGpuVector<Number> &x,
                        const MultiGpuVector<Number> &v);

    // element-wise multiplication
    void scale(const MultiGpuVector<Number> &v);

    // element-wise division
    MultiGpuVector<Number>& operator/=(const MultiGpuVector<Number> &x);

    // element-wise inversion
    MultiGpuVector<Number>& invert();

    // scaled assignment of a vector
    void equ (const Number a,
              const MultiGpuVector<Number> &x);
    // scale the elements of the vector
    // by a fixed value
    MultiGpuVector<Number> & operator *= (const Number a);

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

    void swap(MultiGpuVector<Number> &other);
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


  template <bool atomic=false, typename Number>
  void add_with_indices(MultiGpuVector<Number> &dst,
                        const MultiGpuList<unsigned int> &dst_indices,
                        const MultiGpuList<Number> &src);

  template <typename Number>
  void copy_with_indices(MultiGpuList<Number> &dst,
                         const MultiGpuVector<Number> &src,
                         const MultiGpuList<unsigned int> &src_indices);

  template <typename Number>
  void copy_with_indices(MultiGpuVector<Number> &dst,
                         const MultiGpuList<unsigned int> &dst_indices,
                         const MultiGpuList<Number> &src);


  template <typename DstNumber, typename SrcNumber>
  void copy_with_indices(MultiGpuVector<DstNumber> &dst,
                         const MultiGpuList<unsigned int> &dst_indices,
                         const MultiGpuVector<SrcNumber> &src,
                         const MultiGpuList<unsigned int> &src_indices);
}

#endif /* _MULTI_GPU_VEC_H */
