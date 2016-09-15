/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)cuda_sparse_matrix.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 *
 */

#ifndef dealii__cuda_sparse_matrix_h
#define dealii__cuda_sparse_matrix_h

#include <deal.II/base/subscriptor.h>
#include <deal.II/lac/sparse_matrix.h>
// #include <deal.II/lac/cuda_vector.h>
#include "gpu_vec.h"
#include <cstddef>

// #ifdef DEAL_II_WITH_CUDA

#include <cusparse_v2.h>

// DEAL_II_NAMESPACE_OPEN


#define USE_HYB_MATRIX
// using namespace dealii;


namespace CUDAWrappers
{

  template <typename Number>
  class SparseMatrix : public dealii::Subscriptor {

  private:

    bool initialized;
    unsigned int n_cols;
    unsigned int n_rows;
    int nnz;
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
#ifdef USE_HYB_MATRIX
    cusparseHybMat_t hyb;
#endif
    Number *mat_val;
    int *mat_ptr;
    int *mat_ind;

  public:
    // constructors et al.
    SparseMatrix();

    SparseMatrix(const ::dealii::SparseMatrix<Number> &src_mat);

    ~SparseMatrix();

    void reinit(const ::dealii::SparseMatrix<Number> &src_mat);

    void init();

    unsigned int m() const;

    unsigned int n() const;

    Number el (const unsigned int row,
               const unsigned int col) const {
      ExcNotImplemented();
      return -1000000000000000000;
    }

    void vmult(GpuVector<Number> &dst,
               const GpuVector<Number>  &src) const ;

    void print();

    /**
     * Determine an estimate for the memory consumption (in bytes) of this
     * object. See MemoryConsumption.
     */
    std::size_t memory_consumption () const;

  };

}

// DEAL_II_NAMESPACE_CLOSE

// #endif /* DEAL_II_WITH_CUDA */

#endif /* dealii__cuda_sparse_matrix_h */
