/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#ifndef _LAPLACE_OPERATOR_GPU_H
#define _LAPLACE_OPERATOR_GPU_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>


#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"


using namespace dealii;


//=============================================================================
// coefficient
//=============================================================================

template <int dim>
struct Coefficient
{
  static __host__ double value (const Point<dim> &p){
    // return 1. / (0.05 + 2.*(p.norm_square()));
    return 1.;
  }

  template <typename Number>
  static __device__ Number value (const GpuArray<dim,Number> &p){
    // return 1. / (0.05 + 2.*(p.norm_square()));
    return 1.;
  }

  static __host__  Tensor<1,dim> gradient (const Point<dim> &p){
    // const Tensor<1,dim> dist = -p;
    // const double den = 0.05 + 2.*dist.norm_square();
    // return (4. / (den*den))*dist;
    const Tensor<1,dim> dist = p*0;
    return dist;
  }
};

//=============================================================================
// operator
//=============================================================================

template <int dim, int fe_degree,typename Number>
class LaplaceOperatorGpu : public Subscriptor
{
public:
  LaplaceOperatorGpu ();

  void clear();

  void reinit (const DoFHandler<dim>  &dof_handler,
               const ConstraintMatrix  &constraints);

  unsigned int m () const { return data.n_dofs; }
  unsigned int n () const { return data.n_dofs; }

  void vmult (GpuVector<Number> &dst,
              const GpuVector<Number> &src) const;
  void Tvmult (GpuVector<Number> &dst,
               const GpuVector<Number> &src) const;
  void vmult_add (GpuVector<Number> &dst,
                  const GpuVector<Number> &src) const;
  void Tvmult_add (GpuVector<Number> &dst,
                   const GpuVector<Number> &src) const;

  // we cannot access matrix elements of a matrix free operator directly.
  Number el (const unsigned int row,
             const unsigned int col) const {
    ExcNotImplemented();
    return -1000000000000000000;
  }

  // diagonal for preconditioning
  void set_diagonal (const Vector<Number> &diagonal);
  const GpuVector<Number>& get_diagonal () const {
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_values;
  };

  std::size_t memory_consumption () const;



private:

  void evaluate_coefficient();

  MatrixFreeGpu<dim,Number>   data;
  std::vector<GpuVector<Number > >          coefficient;

  GpuVector<Number>           diagonal_values;
  bool                        diagonal_is_available;


  unsigned int coeff_eval_x_num_blocks;
  unsigned int coeff_eval_y_num_blocks;

};


#define BKSIZE_COEFF_EVAL 128

template <int dim, int fe_degree, typename Number>
LaplaceOperatorGpu<dim,fe_degree,Number>::LaplaceOperatorGpu ()
  :
  Subscriptor()
{}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::clear ()
{
  data.free();
  diagonal_is_available = false;
  diagonal_values.reinit(0);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                  const ConstraintMatrix  &constraints)
{
  typename MatrixFreeGpu<dim,Number>::AdditionalData additional_data;

  additional_data.use_coloring = true;

  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_in_elem;

  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);

  evaluate_coefficient();
}


//  initialize coefficient

template <int dim, int fe_degree, typename Number, typename CoefficientT>
__global__ void local_coeff_eval (Number                                            *coefficient,
                                  const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{
  const unsigned int cell = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

  if(cell < gpu_data.n_cells) {

    const GpuArray<dim,Number> *qpts = gpu_data.quadrature_points;

    for (unsigned int q=0; q<n_q_points; ++q) {
      const unsigned int idx = cell*gpu_data.rowlength + q;
      coefficient[idx] =  CoefficientT::value(qpts[idx]);
    }

  }
}

template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>:: evaluate_coefficient ()
{

  coefficient.resize(data.num_colors);
  for(int c=0; c<data.num_colors; c++) {

    const unsigned int coeff_eval_num_blocks = ceil(data.n_cells[c] / float(BKSIZE_COEFF_EVAL));
    const unsigned int coeff_eval_x_num_blocks = round(sqrt(coeff_eval_num_blocks)); // get closest to even square.
    const unsigned int coeff_eval_y_num_blocks = ceil(double(coeff_eval_num_blocks)/coeff_eval_x_num_blocks);

    const dim3 grid_dim = dim3(coeff_eval_x_num_blocks,coeff_eval_y_num_blocks);
    const dim3 block_dim = dim3(BKSIZE_COEFF_EVAL);

    coefficient[c].resize (data.n_cells[c] * data.get_rowlength());

    local_coeff_eval<dim,fe_degree,Number, Coefficient<dim> >
      <<<grid_dim,block_dim>>> (coefficient[c].getData(),
                               data.get_gpu_data(c));

    CUDA_CHECK_LAST;
  }
}


// multiplication/application functions (symmetric operator -> vmult == Tvmult)

template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult (GpuVector<Number>       &dst,
                                                 const GpuVector<Number> &src) const
{
  dst = 0.0;
  vmult_add (dst, src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult (GpuVector<Number>       &dst,
                                                  const GpuVector<Number> &src) const
{
  dst = 0.0;
  vmult_add (dst,src);
}



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::Tvmult_add (GpuVector<Number>       &dst,
                                                      const GpuVector<Number> &src) const
{
  vmult_add (dst,src);
}

// This is the struct we pass to matrix-free for evaluation on each cell

template <int dim, int fe_degree, typename Number>
struct LocalOperator {
  // coefficient values at this cell
  const Number *coefficient;
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
  static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

  // what to do on each quadrature point
  template <typename FEE>
  __device__ inline void quad_operation(FEE *phi, const unsigned int q,
                                        const unsigned int global_q) const
  {
    phi->submit_gradient (coefficient[global_q] * phi->get_gradient(q), q);
  }

  // what to do fore each cell
  __device__ void apply (Number                          *dst,
                          const Number                    *src,
                          const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                          const unsigned int cell,
                          SharedData<dim,Number> *shdata) const
  {
    FEEvaluationGpu<Number,dim,fe_degree> phi (cell, gpu_data, shdata);

    phi.read_dof_values(src);

    __syncthreads();

    phi.evaluate (false,true);

    // no synch needed since local operation works on 'own' value
    // __syncthreads();

    // apply the local operation above
    phi.apply_quad_point_operations(this);

    __syncthreads();

    phi.integrate (false,true);

    __syncthreads();

    phi.distribute_local_to_global (dst);
  }
};



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult_add (GpuVector<Number>       &dst,
                                                     const GpuVector<Number> &src) const
{
  std::vector <LocalOperator<dim,fe_degree,Number> > loc_op(data.num_colors);
  for(int c=0; c<data.num_colors; c++) {
    loc_op[c].coefficient = coefficient[c].getDataRO();
  }

  data.cell_loop (dst,src,loc_op);

  data.copy_constrained_values(dst,src);

}

// set diagonal (and set values correponding to constrained DoFs to 1)
template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::set_diagonal(const Vector<Number> &diagonal)
{
  AssertDimension (m(), diagonal.size());

  diagonal_values = diagonal;

  data.set_constrained_values(diagonal_values,1.0);

  diagonal_is_available = true;
}



template <int dim, int fe_degree, typename Number>
std::size_t
LaplaceOperatorGpu<dim,fe_degree,Number>::memory_consumption () const
{
  std::size_t apa = (data.memory_consumption () +
                     diagonal_values.memory_consumption() +
                     MemoryConsumption::memory_consumption(diagonal_is_available));
  for(int c=0; c<data.num_colors; c++) {
    apa += coefficient[c].memory_consumption();
  }
  return apa;
}


#endif /* _LAPLACE_OPERATOR_GPU_H */
