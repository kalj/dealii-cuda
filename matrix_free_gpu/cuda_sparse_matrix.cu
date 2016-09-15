
#include "cuda_sparse_matrix.h"

// #ifdef DEAL_II_WITH_CUDA


#define CUSPARSE_CHECK_SUCCESS(call,msg)        \
  do {                                          \
    cusparseStatus_t status = call;             \
    if (status != CUSPARSE_STATUS_SUCCESS) {    \
      fprintf(stderr,"%s\n",msg);               \
      exit(1);                                  \
    }                                           \
  } while(0)


// DEAL_II_NAMESPACE_OPEN

namespace CUDAWrappers
{
  template<typename Number>
  SparseMatrix<Number>::SparseMatrix()
    : initialized(false),n_cols(0), n_rows(0), nnz(0)
  {
    mat_val = NULL;
    mat_ind = NULL;
    mat_ptr = NULL;
  }

  template<typename Number>
  SparseMatrix<Number>::SparseMatrix(const ::dealii::SparseMatrix<Number> &src_mat)
    : initialized(false), n_cols(0), n_rows(0), nnz(0)
  {
    mat_val = NULL;
    mat_ind = NULL;
    mat_ptr = NULL;

    reinit(src_mat);
  }

  template<typename Number>
  SparseMatrix<Number>::~SparseMatrix()
  {
    if(initialized) {

#ifdef USE_HYB_MATRIX

      CUSPARSE_CHECK_SUCCESS(cusparseDestroyHybMat(hyb),"CUSPARSE: Hyb structure destruction failed");

#else
      CudaAssert(cudaFree(mat_val));
      CudaAssert(cudaFree(mat_ind));
      CudaAssert(cudaFree(mat_ptr));
#endif
      CUSPARSE_CHECK_SUCCESS(cusparseDestroyMatDescr(descr),"CUSPARSE: Matrix descriptor destruction failed");
      CUSPARSE_CHECK_SUCCESS(cusparseDestroy(handle),"CUSPARSE: Library release of resources failed");
    }

  }


  template<typename Number>
  void SparseMatrix<Number>::reinit(const ::dealii::SparseMatrix<Number> &src_mat)
  {

    if(!initialized) {
      CUSPARSE_CHECK_SUCCESS(cusparseCreate(&handle),
                             "CUSPARSE: Failed initializing library");
      CUSPARSE_CHECK_SUCCESS(cusparseCreateMatDescr(&descr),
                             "CUSPARSE: Failed initializing matrix descriptor");
      CUSPARSE_CHECK_SUCCESS(cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL),
                             "CUSPARSE: Failed setting matrix type");
      CUSPARSE_CHECK_SUCCESS(cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO),
                             "CUSPARSE: Failed setting matrix index base");

#ifdef USE_HYB_MATRIX
      CUSPARSE_CHECK_SUCCESS(cusparseCreateHybMat(&hyb),"CUSPARSE: Failed initializing hyb structure");
#endif
    }
    else {
#ifndef USE_HYB_MATRIX
      CudaAssert(cudaFree(mat_val));
      CudaAssert(cudaFree(mat_ind));
      CudaAssert(cudaFree(mat_ptr));
#endif
    }

    initialized = true;
    nnz = src_mat.n_nonzero_elements();
    n_rows = src_mat.m();
    n_cols = src_mat.n();

    // initialize CSR matrix from CPU structure
    std::vector<int> Ap(n_rows+1,0);
    std::vector<int> Ai(nnz);
    std::vector<Number> Av(nnz);

    for (int row=0; row<n_rows; ++row) {
      int cursor = Ap[row];
      for (typename ::dealii::SparseMatrix<Number>::const_iterator p=src_mat.begin(row);
           p != src_mat.end(row); ++p) {

        Ai[cursor] = p->column();
        Av[cursor] = p->value();
        ++cursor;

      }
      Ap[row+1] = cursor;

      // This row is now initialized, but the diagonal element is first in the
      // Deal.II world, so we need to resort for CUSPARSE. For simplicity we
      // just make a series of swaps (this is kind of a single run of
      // bubble-sort, which gives us the desired result since the array is
      // already "almost" sorted)

      for(int i = Ap[row];
          (i < Ap[row+1]-1) && (Ai[i] > Ai[i+1]);
          ++i) {
        std::swap (Ai[i], Ai[i+1]);
        std::swap (Av[i], Av[i+1]);
      }

    }

    // allocate device memory
    CudaAssert(cudaMalloc(&mat_ptr,(n_rows+1)*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_ind,nnz*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_val,nnz*sizeof(Number)));

    // copy from host to device
    CudaAssert(cudaMemcpy(mat_ptr,Ap.data(),Ap.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(mat_ind,Ai.data(),Ai.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(mat_val,Av.data(),Av.size()*sizeof(Number),
                          cudaMemcpyHostToDevice));


#ifdef USE_HYB_MATRIX

    // convert to hyb format
    CUSPARSE_CHECK_SUCCESS(cusparseDcsr2hyb(handle,n_rows,n_cols,descr,mat_val,mat_ptr,
                                            mat_ind,hyb,0,CUSPARSE_HYB_PARTITION_AUTO),
                           "CUSPARSE: Failed converting matrix to hyb format");


    CudaAssert(cudaFree(mat_val));
    CudaAssert(cudaFree(mat_ind));
    CudaAssert(cudaFree(mat_ptr));
#endif
  }

  template<typename Number>
  void SparseMatrix<Number>::init()
  {

    if(!initialized) {
      CUSPARSE_CHECK_SUCCESS(cusparseCreate(&handle),
                             "CUSPARSE: Failed initializing library");
      CUSPARSE_CHECK_SUCCESS(cusparseCreateMatDescr(&descr),
                             "CUSPARSE: Failed initializing matrix descriptor");
      CUSPARSE_CHECK_SUCCESS(cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL),
                             "CUSPARSE: Failed setting matrix type");
      CUSPARSE_CHECK_SUCCESS(cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO),
                             "CUSPARSE: Failed setting matrix index base");

#ifdef USE_HYB_MATRIX
      CUSPARSE_CHECK_SUCCESS(cusparseCreateHybMat(&hyb),"CUSPARSE: Failed initializing hyb structure");
#endif
    }
    else {
#ifndef USE_HYB_MATRIX
      CudaAssert(cudaFree(mat_val));
      CudaAssert(cudaFree(mat_ind));
      CudaAssert(cudaFree(mat_ptr));
#endif
    }

    initialized = true;



    /* Initialize with this matrix:


       [ 1      0       1       0
         1      1       0       1
         0      1       0       0
         0      0       1       1]

    */



    n_rows = 4;
    n_cols = 4;

    std::vector<std::vector<Number>> vals(n_rows);
    std::vector<std::vector<int>> inds(n_rows);

    vals[0].push_back(1.0); // A[0,0] = 1.0
    inds[0].push_back(0);

    vals[0].push_back(1.0); // A[0,2] = 1.0
    inds[0].push_back(2);

    vals[1].push_back(1.0); // A[1,0] = 1.0
    inds[1].push_back(0);

    vals[1].push_back(1.0); // A[1,1] = 1.0
    inds[1].push_back(1);

    vals[1].push_back(1.0); // A[1,3] = 1.0
    inds[1].push_back(3);

    vals[2].push_back(1.0); // A[2,1] = 1.0
    inds[2].push_back(1);

    vals[3].push_back(1.0); // A[3,2] = 1.0
    inds[3].push_back(2);

    vals[3].push_back(1.0); // A[3,3] = 1.0
    inds[3].push_back(3);


    // initialize CSR matrix from CPU structure
    std::vector<int> Ap(n_rows+1,0);
    std::vector<int> Ai;
    std::vector<Number> Av;

    for (int row=0; row<n_rows; ++row) {

      Av.insert( Av.end(), vals[row].begin(), vals[row].end() );
      Ai.insert( Ai.end(), inds[row].begin(), inds[row].end() );
      Ap[row+1] = Ap[row]+vals[row].size();
    }
    nnz = Ap[n_rows];

    CudaAssert(cudaMalloc(&mat_ptr,(n_rows+1)*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_ind,nnz*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_val,nnz*sizeof(Number)));

    // copy from host to device
    CudaAssert(cudaMemcpy(mat_ptr,Ap.data(),Ap.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(mat_ind,Ai.data(),Ai.size()*sizeof(int),
                          cudaMemcpyHostToDevice));
    CudaAssert(cudaMemcpy(mat_val,Av.data(),Av.size()*sizeof(Number),
                          cudaMemcpyHostToDevice));


#ifdef USE_HYB_MATRIX

    // convert to hyb format
    CUSPARSE_CHECK_SUCCESS(cusparseDcsr2hyb(handle,n_rows,n_cols,descr,mat_val,mat_ptr,
                                            mat_ind,hyb,0,CUSPARSE_HYB_PARTITION_AUTO),
                           "CUSPARSE: Failed converting matrix format from csr to hyb");


    CudaAssert(cudaFree(mat_val));
    CudaAssert(cudaFree(mat_ind));
    CudaAssert(cudaFree(mat_ptr));
#endif
  }



  template<typename Number>
  unsigned int SparseMatrix<Number>::m() const
  {
    return n_rows;
  }

  template<typename Number>
  unsigned int SparseMatrix<Number>::n() const
  {
    return n_cols;
  }


  template<typename Number>
  void SparseMatrix<Number>::vmult(GpuVector<Number> &dst,
                                   const GpuVector<Number>  &src) const
  {
    const Number zero = 0.0;
    const Number one = 1.0;
    Number *dst_buf = dst.getData();
    const Number *vec_buf = src.getDataRO();

#ifdef USE_HYB_MATRIX
    CUSPARSE_CHECK_SUCCESS(cusparseDhybmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          &one, descr, hyb, vec_buf, &zero, dst_buf),
                           "CUSPARSE: hybmv failed");
#else
    CUSPARSE_CHECK_SUCCESS(cusparseDcsrmv(handle,CUSPARSE_OPERATION_NON_TRANSPOSE, n_rows, n_cols, nnz,
                                          &one, descr, mat_val, mat_ptr, mat_ind,
                                          vec_buf, &zero, dst_buf),
                           "CUSPARSE: csrmv failed");
#endif

  }


  template<typename Number>
  void SparseMatrix<Number>::print()
  {
#ifdef USE_HYB_MATRIX
    // allocate device memory
    CudaAssert(cudaMalloc(&mat_ptr,(n_rows+1)*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_ind,nnz*sizeof(int)));
    CudaAssert(cudaMalloc(&mat_val,nnz*sizeof(Number)));


    // convert to hyb format
    CUSPARSE_CHECK_SUCCESS(cusparseDhyb2csr(handle,descr,hyb,mat_val,mat_ptr,
                                            mat_ind),
                           "CUSPARSE: Failed converting matrix format from hyb to csr");

#endif

    // initialize CSR matrix from CPU structure
    std::vector<int> Ap(n_rows+1);
    std::vector<int> Ai(nnz);
    std::vector<Number> Av(nnz);

    // copy from host to device
    CudaAssert(cudaMemcpy(Ap.data(),mat_ptr,Ap.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CudaAssert(cudaMemcpy(Ai.data(),mat_ind,Ai.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CudaAssert(cudaMemcpy(Av.data(),mat_val,Av.size()*sizeof(Number), cudaMemcpyDeviceToHost));

    for(int i=0; i<n_rows; i++) {
      for (int j=Ap[i]; j < Ap[i+1]; j++) {
        printf("[%d,%d]: %g\n",i,Ai[j],Av[j]);
      }
    }

#ifdef USE_HYB_MATRIX
    CudaAssert(cudaFree(mat_val));
    CudaAssert(cudaFree(mat_ind));
    CudaAssert(cudaFree(mat_ptr));
#endif

  }



  template <typename number>
  std::size_t
  SparseMatrix<number>::memory_consumption () const
  {
    return nnz*static_cast<std::size_t>(sizeof(number) + sizeof(int))
      + (n_cols+1)*static_cast<std::size_t>(sizeof(int))
      + sizeof(*this);
  }

}


template
class CUDAWrappers::SparseMatrix<double>;

// DEAL_II_NAMESPACE_CLOSE

// #endif /* DEAL_II_WITH_CUDA */