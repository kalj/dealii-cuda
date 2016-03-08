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



#ifdef MATRIX_FREE_PAR_IN_ELEM
#define MATRIX_FREE_BKSIZE_APPLY_DEF 1
#else
#define MATRIX_FREE_BKSIZE_APPLY_DEF 32
#endif

#ifndef MATRIX_FREE_BKSIZE_APPLY
#define MATRIX_FREE_BKSIZE_APPLY (MATRIX_FREE_BKSIZE_APPLY_DEF)
#endif


using namespace dealii;

// forward declaration
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
  unsigned int n_constrained_dofs;
  unsigned int dofs_per_cell;
  unsigned int qpts_per_cell;

  bool use_coloring;

  unsigned int num_colors;

  enum ParallelizationScheme {scheme_par_in_elem, scheme_par_over_elems};

  struct AdditionalData
  {

    AdditionalData (const ParallelizationScheme parallelization_scheme = scheme_par_in_elem,
                    const bool         use_coloring = false,
                    const UpdateFlags  mapping_update_flags  = update_gradients | update_JxW_values
                    )
      :
      parallelization_scheme(parallelization_scheme),
      use_coloring (use_coloring),
      mapping_update_flags (mapping_update_flags)
    {};

    ParallelizationScheme parallelization_scheme;

    bool                use_coloring;
    unsigned int        num_colors;

    UpdateFlags         mapping_update_flags;
  };


private:

  ParallelizationScheme parallelization_scheme;

  // stuff related to FE ...

  // ...and quadrature
  std::vector<point_type*> quadrature_points;

  // mapping and mesh info
  // (could be made into a mapping_info struct)
  std::vector<unsigned int*>    loc2glob;
  std::vector<Number*>          inv_jac;
  std::vector<Number*>          JxW;

  // constraints
  unsigned int *constrained_dofs;
  std::vector<unsigned int*>    constraint_mask;

  // GPU kernel parameters
  std::vector<dim3> grid_dim;
  std::vector<dim3> block_dim;

  dim3 constr_grid_dim;
  dim3 constr_block_dim;
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

  };


  MatrixFreeGpu()
    : constrained_dofs(NULL),
       use_coloring(false)
  {}

  void reinit(const Mapping<dim>        &mapping,
              const DoFHandler<dim>     &dof_handler,
              const ConstraintMatrix    &constraints,
              // const IndexSet         &locally_owned_dofs,
              const Quadrature<1>       &quad,
              const AdditionalData      additional_data = AdditionalData());



  void reinit(const DoFHandler<dim>     &dof_handler,
              const ConstraintMatrix    &constraints,
              // const IndexSet         &locally_owned_dofs,
              const Quadrature<1>       &quad,
              const AdditionalData      additional_data = AdditionalData())
  {
    MappingQ1<dim>  mapping;
    reinit(mapping,dof_handler,constraints,quad,additional_data);
  }

  const GpuData get_gpu_data(unsigned int color) const {
    GpuData data;
    data.quadrature_points = quadrature_points[color];
    data.loc2glob = loc2glob[color];
    data.inv_jac = inv_jac[color];
    data.JxW = JxW[color];
    data.constraint_mask = constraint_mask[color];
    data.n_cells = n_cells[color];
    data.use_coloring = use_coloring;
    return data;
  }

  template <typename LocOp>
  void cell_loop_shmem(GpuVector<Number> &dst, const GpuVector<Number> &src,  const std::vector<LocOp> &loc_op) const;


  template <typename LocOp>
  void cell_loop_pmem(GpuVector<Number> &dst, const GpuVector<Number> &src,  const std::vector<LocOp> &loc_op) const;


  void copy_constrained_values(GpuVector <Number> &dst, const GpuVector<Number> &src) const;
  void set_constrained_values(GpuVector <Number> &dst, Number val) const;

  void free();

  std::size_t memory_consumption() const { return 1; }


  friend class ReinitHelper<dim,Number>;
};


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


template <typename LocOp,int dim, typename Number>
__global__ void apply_kernel_shmem (Number                          *dst,
                                    const Number                    *src,
                                    const LocOp                    loc_op,
                                    const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{

  __shared__ Number values[MATRIX_FREE_BKSIZE_APPLY*LocOp::n_local_dofs];
  __shared__ Number gradients[dim][MATRIX_FREE_BKSIZE_APPLY*LocOp::n_q_points];

  const unsigned int local_cell = (threadIdx.x/LocOp::n_dofs_1d);
  const unsigned int cell = local_cell + MATRIX_FREE_BKSIZE_APPLY*(blockIdx.x+gridDim.x*blockIdx.y);

  Number *gq[dim];
  for(int d = 0; d < dim; ++d) gq[d] = &gradients[d][local_cell*LocOp::n_q_points];

  SharedData<dim,Number> shdata(&values[local_cell*LocOp::n_local_dofs],
                                gq);

  if(cell < gpu_data.n_cells) {
    loc_op.apply(dst,src,&gpu_data,cell,&shdata);
  }
}


template <typename LocOp,int dim, typename Number>
__global__ void apply_kernel_pmem (Number                          *dst,
                              const Number                    *src,
                              const LocOp                    loc_op,
                              const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{

  const unsigned int cell = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);

  if(cell < gpu_data.n_cells) {
    loc_op.apply(dst,src,&gpu_data,cell);
  }
}


template <int dim, typename Number>
template <typename LocOp>
void MatrixFreeGpu<dim,Number>::cell_loop_shmem(GpuVector<Number> &dst, const GpuVector<Number> &src,
                                          const std::vector<LocOp> &loc_op) const
{
  for(int c = 0; c < num_colors; ++c) {

    apply_kernel_shmem<LocOp,dim,Number> <<<grid_dim[c],block_dim[c]>>> (dst.getData(), src.getDataRO(),
                                                                         loc_op[c], get_gpu_data(c));
    CUDA_CHECK_LAST;
  }
}

template <int dim, typename Number>
template <typename LocOp>
void MatrixFreeGpu<dim,Number>::cell_loop_pmem(GpuVector<Number> &dst, const GpuVector<Number> &src,
                                          const std::vector<LocOp> &loc_op) const
{
  for(int c = 0; c < num_colors; ++c) {

    apply_kernel_pmem<LocOp,dim,Number> <<<grid_dim[c],block_dim[c]>>> (dst.getData(), src.getDataRO(),
                                                                         loc_op[c], get_gpu_data(c));
    CUDA_CHECK_LAST;
  }
}

#include "matrix_free_gpu.cu"

#endif /* _MATRIX_FREE_GPU_H */
