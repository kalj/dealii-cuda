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


//-------------------------------------------------------------------------
//  coefficient
//-------------------------------------------------------------------------

template <int dim, typename Number>
struct Coefficient
{
  __device__ static Number value (const GpuArray<dim,Number> &p){

    return 1. / (0.05 + 2.*(p.norm_square()));
  }

  __device__ static void value_list (const GpuArray<dim,Number> *points,
                                     Number                     *values,
                                     const unsigned int         len) {
    for (unsigned int i=0; i<len; ++i)
      values[i] = value(points[i]);
  }
};

// coefficient for cpu


template <int dim>
class Coeff : public Function<dim>
{
public:
  Coeff ()  : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  template <typename number>
  number value (const Point<dim,number> &p,
                const unsigned int component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};


template <int dim>
template <typename number>
number Coeff<dim>::value (const Point<dim,number> &p,
                          const unsigned int /*component*/) const
{
  return 1.0 / (0.05 + 2.0*p.square());
}



template <int dim>
double Coeff<dim>::value (const Point<dim>  &p,
                          const unsigned int component) const
{
  return value<double>(p,component);
}



template <int dim>
void Coeff<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value<double>(points[i],component);
}


//-------------------------------------------------------------------------
// operator
//-------------------------------------------------------------------------



template <int dim, int fe_degree,typename Number>
class LaplaceOperatorGpu : public Subscriptor
{
public:
  LaplaceOperatorGpu ();

  void clear();

  void reinit (const DoFHandler<dim>  &dof_handler,
               const ConstraintMatrix  &constraints);

  unsigned int m () const;
  unsigned int n () const;

  void vmult (GpuVector<Number> &dst,
              const GpuVector<Number> &src) const;
  void Tvmult (GpuVector<Number> &dst,
               const GpuVector<Number> &src) const;
  void vmult_add (GpuVector<Number> &dst,
                  const GpuVector<Number> &src) const;
  void Tvmult_add (GpuVector<Number> &dst,
                   const GpuVector<Number> &src) const;

  Number el (const unsigned int row,
             const unsigned int col) const {
    Assert (false, ExcNotImplemented());
    return -1000000000000000000;
  }
  void set_diagonal (const Vector<Number> &diagonal);

  const GpuVector<Number>& get_diagonal () const {
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_values;
  };

  std::size_t memory_consumption () const;



#ifdef FINE_GRAIN_TIMER
  float *t_read;
  float *t_write;
  float *t_comp;
  unsigned int max_ncells;
  void print_fine_grain_timers() const;
#endif

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
unsigned int
LaplaceOperatorGpu<dim,fe_degree,Number>::m () const
{
  return data.n_dofs;
}



template <int dim, int fe_degree, typename Number>
unsigned int
LaplaceOperatorGpu<dim,fe_degree,Number>::n () const
{
  return data.n_dofs;
}



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

#ifdef MATRIX_FREE_COLOR
  additional_data.use_coloring = true;
#else
  additional_data.use_coloring = false;
#endif

#ifdef MATRIX_FREE_PAR_IN_ELEM
  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_in_elem;
#else
  additional_data.parallelization_scheme = MatrixFreeGpu<dim,Number>::scheme_par_over_elems;
#endif


  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);


#ifdef FINE_GRAIN_TIMER
  max_ncells = 0;
  for(unsigned int c=0; c < data.num_colors; c++) {
    if(data.n_cells[c] > max_ncells)
      max_ncells = data.n_cells[c];
  }

  CUDA_CHECK_SUCCESS(cudaMalloc(&t_read, max_ncells*sizeof(float)));
  CUDA_CHECK_SUCCESS(cudaMalloc(&t_write, max_ncells*sizeof(float)));
  CUDA_CHECK_SUCCESS(cudaMalloc(&t_comp, max_ncells*sizeof(float)));
#endif


  evaluate_coefficient();
}



template <int dim, int fe_degree, typename Number, typename coefficient_function>
__global__ void local_coeff_eval (Number                          *coeff,
                                  const typename MatrixFreeGpu<dim,Number>::GpuData gpu_data)
{
  const unsigned int cell = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

  if(cell < gpu_data.n_cells) {

    const GpuArray<dim,Number> *qpts = gpu_data.quadrature_points;

    for (unsigned int q=0; q<n_q_points; ++q) {
#ifdef MATRIX_FREE_PAR_IN_ELEM
      const unsigned int idx = cell*n_q_points + q;
#else
      const unsigned int idx = cell + gpu_data.n_cells*q;
#endif
      coeff[idx] =  coefficient_function::value(qpts[idx]);
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

    coefficient[c].resize (data.n_cells[c] * data.qpts_per_cell);

    local_coeff_eval<dim,fe_degree,Number,Coefficient<dim,Number> > <<<grid_dim,block_dim>>>(coefficient[c].getData(),
                                                                                             data.get_gpu_data(c));
    CUDA_CHECK_LAST;
  }
}





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


template <int dim, int fe_degree, typename Number>
struct LocalOperator {
  const Number *coefficient;
  static const unsigned int n_dofs_1d = fe_degree+1;
  static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
  static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;
#ifdef FINE_GRAIN_TIMER
  float *t_read;
  float *t_write;
  float *t_comp;
#endif

  template <typename FEE>
  __device__ inline void quad_operation(FEE *phi, const unsigned int q,const unsigned int global_q) const
  {
    phi->submit_gradient (coefficient[global_q] * phi->get_gradient(q), q);
  }

  __device__ void apply (Number                          *dst,
                          const Number                    *src,
                          const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
#ifdef MATRIX_FREE_PAR_IN_ELEM
                          const unsigned int cell,
                          SharedData<dim,Number> *shdata) const
#else
                          const unsigned int cell) const
#endif
  {
#ifdef FINE_GRAIN_TIMER

#ifdef MATRIX_FREE_PAR_IN_ELEM
    const bool flag = threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0;
#else
    const bool flag = true;
#endif

    clock_t start, stop;
#endif

#ifdef MATRIX_FREE_PAR_IN_ELEM
    FEEvaluationGpuPIE<Number,dim,fe_degree> phi (cell, gpu_data, shdata);
#else
    FEEvaluationGpu<Number,dim,fe_degree> phi (cell, gpu_data);
#endif

#ifdef FINE_GRAIN_TIMER
    if(flag) {
      start = clock();
    }
#endif

    phi.read_dof_values(src);

#ifdef MATRIX_FREE_PAR_IN_ELEM
    __syncthreads();
#endif

#ifdef FINE_GRAIN_TIMER
    if(flag) {
      stop = clock();
      t_read[cell] += stop - start;

      start = clock();
    }
#endif

    phi.evaluate (false,true,false);
#ifdef MATRIX_FREE_PAR_IN_ELEM
    // __syncthreads();
#endif

    // apply the local operation above
    phi.apply_quad_point_operations(this);

#ifdef MATRIX_FREE_PAR_IN_ELEM
    __syncthreads();
#endif

    phi.integrate (false,true);
#ifdef MATRIX_FREE_PAR_IN_ELEM
    __syncthreads();
#endif

#ifdef FINE_GRAIN_TIMER
    if(flag) {
      stop = clock();
      t_comp[cell] += stop - start;

      start = clock();
    }
#endif

    phi.distribute_local_to_global (dst);

#ifdef FINE_GRAIN_TIMER
    if(flag) {

      stop = clock();
      t_write[cell] += stop - start;
    }
#endif
  }
};



template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::vmult_add (GpuVector<Number>       &dst,
                                                     const GpuVector<Number> &src) const
{

#ifdef FINE_GRAIN_TIMER
  CUDA_CHECK_SUCCESS(cudaMemset(t_read, 0, max_ncells*sizeof(float)));
  CUDA_CHECK_SUCCESS(cudaMemset(t_write, 0, max_ncells*sizeof(float)));
  CUDA_CHECK_SUCCESS(cudaMemset(t_comp, 0, max_ncells*sizeof(float)));
#endif

  std::vector <LocalOperator<dim,fe_degree,Number> > loc_op(data.num_colors);
  for(int c=0; c<data.num_colors; c++) {
    loc_op[c].coefficient = coefficient[c].getDataRO();
#ifdef FINE_GRAIN_TIMER
    loc_op[c].t_read = t_read;
    loc_op[c].t_write = t_write;
    loc_op[c].t_comp = t_comp;
#endif
  }

#ifdef MATRIX_FREE_PAR_IN_ELEM
  data.cell_loop_shmem (dst,src,loc_op);
#else
  data.cell_loop_pmem (dst,src,loc_op);
#endif

  data.copy_constrained_values(dst,src);

}

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

#ifdef FINE_GRAIN_TIMER

void compute_stats(float &mean, float &var, float &max, const float *arr_dev, const int n) {

  float *arr = new float[n];
  CUDA_CHECK_SUCCESS(cudaMemcpy(arr,arr_dev,n*sizeof(float),
                                cudaMemcpyDeviceToHost));

  const float freq = 1100e6;

  float sum = 0;
  float sqsum = 0;
  max = 0;

  for(int i = 0; i < n; ++i) {
    max = arr[i] > max ? arr[i] : max;

    sum += arr[i]/freq*1000;
    sqsum += arr[i]*arr[i]/freq*1000/freq*1000;

  }

  mean = sum / n;
  var = sqsum/n - mean*mean;
  max = max/freq*1000;

  delete[] arr;
}

template <int dim, int fe_degree, typename Number>
void
LaplaceOperatorGpu<dim,fe_degree,Number>::print_fine_grain_timers () const
{
  float read_mean, read_var, read_max, write_mean, write_var,
    write_max, comp_mean, comp_var, comp_max;
  compute_stats(read_mean,read_var,read_max,t_read,max_ncells);
  compute_stats(write_mean,write_var,write_max,t_write,max_ncells);
  compute_stats(comp_mean,comp_var,comp_max,t_comp,max_ncells);


  // printf("Read: %g ±%g ms\n",
  //        read_mean,
  //        sqrt(read_var));
  // printf("Write: %g ±%g ms\n",
  //        write_mean,
  //        sqrt(write_var));
  // printf("Compute: %g ±%g ms\n",
  //        comp_mean,
  //        sqrt(comp_var));


  float tot = read_mean + write_mean + comp_mean;

  printf("Read: %g (%.2g %%)\n",
         read_mean,
         100*read_mean/tot);
  printf("Write: %g (%.2g %%)\n",
         write_mean,
         write_mean*100/tot);
  printf("Compute: %g (%.2g %%)\n",
         comp_mean,
         comp_mean*100/tot);
}
#endif


#endif /* _LAPLACE_OPERATOR_GPU_H */
