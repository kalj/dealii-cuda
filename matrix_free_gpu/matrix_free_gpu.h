/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)matrix_free_gpu.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _MATRIX_FREE_GPU_H
#define _MATRIX_FREE_GPU_H

#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include "defs.h"
#include "utils.h"
#include "gpu_array.cuh"
#include "gpu_vec.h"
#include "maybecuda.h"
#include "cuda_utils.cuh"


//=============================================================================
// This implements a class for matrix-free computations on the GPU. It
// essentially consists of the following functions:
//
// - reinit:    sets up the object with a new grid, usually after refinement
// - cell_loop: performs the cell-local operations for all elements in
//              parallel. the local operation to perform is passed as a
//              read-only struct with a function 'apply' and, possibly some
//              data, such as a cell-local coefficient values.
//=============================================================================



using namespace dealii;

// forward declaration of initialization help class
template <int dim, typename Number>
class ReinitHelper;


__constant__ double shape_values[(MAX_ELEM_DEGREE+1)*(MAX_ELEM_DEGREE+1)];
__constant__ double shape_gradient[(MAX_ELEM_DEGREE+1)*(MAX_ELEM_DEGREE+1)];


template <int dim, typename Number>
class MatrixFreeGpu {
public:
  typedef GpuArray<dim,GpuArray<dim,Number> > jac_type;
  typedef GpuArray<dim,Number> point_type;
  std::vector<unsigned int> n_cells;
  unsigned int fe_degree;
  unsigned int n_cells_tot;
  unsigned int n_dofs;
  unsigned int dofs_per_cell;
  unsigned int qpts_per_cell;

  bool use_coloring;

  unsigned int num_colors;

  enum ParallelizationScheme {scheme_par_in_elem, scheme_par_over_elems};

  struct AdditionalData
  {

    AdditionalData (const ParallelizationScheme parallelization_scheme = scheme_par_in_elem,
                    const bool                  use_coloring = false,
                    const UpdateFlags           mapping_update_flags  = update_gradients | update_JxW_values,
                    const unsigned int          level_mg_handler = numbers::invalid_unsigned_int
                    )
      :
      parallelization_scheme(parallelization_scheme),
      use_coloring (use_coloring),
      mapping_update_flags (mapping_update_flags),
      level_mg_handler(level_mg_handler)
    {};

    ParallelizationScheme parallelization_scheme;

    bool                use_coloring;
    unsigned int        num_colors;

    UpdateFlags         mapping_update_flags;

    unsigned int        level_mg_handler;
  };


private:

  ParallelizationScheme parallelization_scheme;

  /**
   * This option can be used to define whether we work on a certain level of the
   * mesh, and not the active cells. If set to invalid_unsigned_int (which is
   * the default value), the active cells are gone through, otherwise the level
   * given by this parameter.
   */
  unsigned int             level_mg_handler;

  // stuff related to FE ...

  // ...and quadrature
  std::vector<point_type*> quadrature_points;

  // mapping and mesh info
  // (could be made into a mapping_info struct)
  std::vector<unsigned int*>    loc2glob;
  std::vector<Number*>          inv_jac;
  std::vector<Number*>          JxW;

  // constraints
  std::vector<unsigned int*>    constraint_mask;

  // GPU kernel parameters
  std::vector<dim3> grid_dim;
  std::vector<dim3> block_dim;

  // related to parallelization
  unsigned int cells_per_block;

  // for alignment
  unsigned int rowlength;
  std::vector<unsigned int> rowstart;

public:

  struct GpuData {

    // ...and quadrature
    point_type    *quadrature_points;

    // mapping and mesh info
    // (could be made into a mapping_info struct)
    unsigned int   *loc2glob;
    Number         *inv_jac;
    Number         *JxW;

    unsigned int   *constraint_mask;

    unsigned int n_cells;

    bool use_coloring;

    unsigned int rowlength;

    unsigned int rowstart;
  };


  MatrixFreeGpu();

  unsigned int get_rowlength() const;

  void reinit(const Mapping<dim>        &mapping,
              const DoFHandler<dim>     &dof_handler,
              const ConstraintMatrix    &constraints,
              const Quadrature<1>       &quad,
              const AdditionalData      additional_data = AdditionalData());



  void reinit(const DoFHandler<dim>     &dof_handler,
              const ConstraintMatrix    &constraints,
              const Quadrature<1>       &quad,
              const AdditionalData      additional_data = AdditionalData());

  GpuData get_gpu_data(unsigned int color) const;

  // apply the local operation on each element in parallel. loc_op is a vector
  // with one entry for each color. That is usually the same operator, but with
  // different local data (e.g. coefficient values).
  template <typename LocOp>
  void cell_loop(GpuVector<Number> &dst, const GpuVector<Number> &src,
                 const LocOp &loc_op) const;

  // same but for only a single vector (the destination)
  template <typename LocOp>
  void cell_loop(GpuVector<Number> &dst,
                 const LocOp &loc_op) const;

  void free();

  template <typename Op>
  void evaluate_on_cells(GpuVector<Number> &v) const;

  std::size_t memory_consumption() const;

  friend class ReinitHelper<dim,Number>;
};


//=============================================================================
// implementations
//=============================================================================


template <int dim, typename Number>
MatrixFreeGpu<dim,Number>::MatrixFreeGpu()
  : use_coloring(false),
    rowlength(0),
    level_mg_handler(numbers::invalid_unsigned_int)
{}


template <int dim, typename Number>
unsigned int MatrixFreeGpu<dim,Number>::get_rowlength() const
{
  return rowlength;
}

template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::reinit(const DoFHandler<dim>     &dof_handler,
                                       const ConstraintMatrix    &constraints,
                                       const Quadrature<1>       &quad,
                                       const AdditionalData      additional_data)
{
  MappingQ1<dim>  mapping;
  reinit(mapping,dof_handler,constraints,quad,additional_data);
}

template <int dim, typename Number>
MatrixFreeGpu<dim,Number>::GpuData
MatrixFreeGpu<dim,Number>::get_gpu_data(unsigned int color) const
{
  MatrixFreeGpu<dim,Number>::GpuData data;
  data.quadrature_points = quadrature_points[color];
  data.loc2glob = loc2glob[color];
  data.inv_jac = inv_jac[color];
  data.JxW = JxW[color];
  data.constraint_mask = constraint_mask[color];
  data.n_cells = n_cells[color];
  data.use_coloring = use_coloring;
  data.rowlength = rowlength;
  data.rowstart = rowstart[color];
  return data;
}

// Struct to pass the shared memory into a general user function
template <int dim, typename Number>
struct SharedData {
  __device__ SharedData(Number *vd,
                        Number *gq[dim])
    : values(vd)
  {
    for(int d = 0; d < dim; ++d) {
      gradients[d] = gq[d];
    }
  }

  Number             *values;
  Number             *gradients[dim];
};

// this function determines the number of cells per block, possibly at compile
// time
__host__ __device__ constexpr unsigned int cells_per_block_shmem(int dim,
                                                                 int fe_degree)
{
#ifdef MATRIX_FREE_CELLS_PER_BLOCK
  return MATRIX_FREE_CELLS_PER_BLOCK;
#else
  return dim==2 ? (fe_degree==1 ? 32
                   : fe_degree==2 ? 8
                   : fe_degree==3 ? 4
                   : fe_degree==4 ? 4
                   : 0) :
    dim==3 ? (fe_degree==1 ? 8
              : fe_degree==2 ? 2
              : fe_degree==3 ? 1
              : fe_degree==4 ? 1
              : 0) : 0;
#endif
}


template <typename LocOp,int dim, typename Number>
__global__ void apply_kernel_shmem (Number                          *dst,
                                    const Number                    *src,
                                    const LocOp                    loc_op,
                                    const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{
  const unsigned int cells_per_block = cells_per_block_shmem(dim,LocOp::n_dofs_1d-1);

  // TODO: make use of dynamically allocated shared memory to avoid this mess.
  __shared__ Number values[cells_per_block*LocOp::n_local_dofs];
  __shared__ Number gradients[dim][cells_per_block*LocOp::n_q_points];

  const unsigned int local_cell = (threadIdx.x/LocOp::n_dofs_1d);
  const unsigned int cell = local_cell + cells_per_block*(blockIdx.x+gridDim.x*blockIdx.y);

  Number *gq[dim];
  for(int d = 0; d < dim; ++d) gq[d] = &gradients[d][local_cell*LocOp::n_q_points];

  SharedData<dim,Number> shdata(&values[local_cell*LocOp::n_local_dofs],gq);

  if(cell < gpu_data.n_cells) {
    loc_op.cell_apply(dst,src,&gpu_data,cell,&shdata);
  }
}

template <typename LocOp,int dim, typename Number>
__global__ void apply_kernel_shmem (Number                          *dst,
                                    const LocOp                    loc_op,
                                    const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{
  const unsigned int cells_per_block = cells_per_block_shmem(dim,LocOp::n_dofs_1d-1);

  // TODO: make use of dynamically allocated shared memory to avoid this mess.
  __shared__ Number values[cells_per_block*LocOp::n_local_dofs];
  __shared__ Number gradients[dim][cells_per_block*LocOp::n_q_points];

  const unsigned int local_cell = (threadIdx.x/LocOp::n_dofs_1d);
  const unsigned int cell = local_cell + cells_per_block*(blockIdx.x+gridDim.x*blockIdx.y);

  Number *gq[dim];
  for(int d = 0; d < dim; ++d) gq[d] = &gradients[d][local_cell*LocOp::n_q_points];

  SharedData<dim,Number> shdata(&values[local_cell*LocOp::n_local_dofs],gq);

  if(cell < gpu_data.n_cells) {
    loc_op.cell_apply(dst,&gpu_data,cell,&shdata);
  }
}



template <int dim, typename Number>
template <typename LocOp>
void MatrixFreeGpu<dim,Number>::cell_loop(GpuVector<Number> &dst, const GpuVector<Number> &src,
                                          const LocOp &loc_op) const
{
  for(int c = 0; c < num_colors; ++c) {

    apply_kernel_shmem<LocOp,dim,Number> <<<grid_dim[c],block_dim[c]>>> (dst.getData(), src.getDataRO(),
                                                                         loc_op, get_gpu_data(c));
    CUDA_CHECK_LAST;
  }
}

template <int dim, typename Number>
template <typename LocOp>
void MatrixFreeGpu<dim,Number>::cell_loop(GpuVector<Number> &dst,
                                          const LocOp &loc_op) const
{
  for(int c = 0; c < num_colors; ++c) {

    apply_kernel_shmem<LocOp,dim,Number> <<<grid_dim[c],block_dim[c]>>> (dst.getData(),
                                                                         loc_op, get_gpu_data(c));
    CUDA_CHECK_LAST;
  }
}



template <int dim, typename Number, typename Op>
__global__ void cell_eval_kernel (Number                                            *vec,
                                  const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{
  const unsigned int cell = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);

  if(cell < gpu_data.n_cells) {

    const GpuArray<dim,Number> *qpts = gpu_data.quadrature_points;
    Op::eval(vec + cell*gpu_data.rowlength,
             qpts + cell*gpu_data.rowlength);

  }
}


#define BKSIZE_COEFF_EVAL 128

template <int dim, typename Number>
template <typename Op>
void MatrixFreeGpu<dim,Number>::evaluate_on_cells(GpuVector<Number> &vec) const
{
  vec.resize (n_cells_tot * rowlength);

  for(int c = 0; c < num_colors; ++c) {

    const unsigned int num_blocks = ceil(n_cells[c] / float(BKSIZE_COEFF_EVAL));
    const unsigned int num_blocks_x = round(sqrt(num_blocks)); // get closest to even square.
    const unsigned int num_blocks_y = ceil(double(num_blocks)/num_blocks_x);

    const dim3 grid_dim = dim3(num_blocks_x,num_blocks_y);
    const dim3 block_dim = dim3(BKSIZE_COEFF_EVAL);

    cell_eval_kernel<dim,Number,Op> <<<grid_dim,block_dim>>> (vec.getData() + rowstart[c],
                                                              get_gpu_data(c));
    CUDA_CHECK_LAST;

  }
}

template <int dim, typename Number>
std::size_t MatrixFreeGpu<dim,Number>::memory_consumption() const
{
  std::size_t bytes = n_cells.size()*sizeof(unsigned int)*(2) // n_cells and rowstarts
    + 2*num_colors*sizeof(dim3);                               // kernel launch parameters

  for(int c = 0; c < num_colors; ++c) {
    bytes +=
      n_cells[c]*rowlength*sizeof(unsigned int)     // loc2glob
      + n_cells[c]*rowlength*dim*dim*sizeof(Number) // inv_jac
      + n_cells[c]*rowlength*sizeof(Number)         // JxW
      + n_cells[c]*rowlength*sizeof(point_type)     // quadrature_points
      + n_cells[c]*sizeof(unsigned int);            // constraint_mask
  }

  return bytes;
}

#include "matrix_free_gpu.cu"

#endif /* _MATRIX_FREE_GPU_H */
