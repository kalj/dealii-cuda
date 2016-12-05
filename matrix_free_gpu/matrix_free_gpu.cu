/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)matrix_free_gpu.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/base/graph_coloring.h>

#include "coloring.h"

#ifdef MATRIX_FREE_HANGING_NODES
#include "hanging_nodes.cuh"
#endif

#include "cuda_utils.cuh"


//=============================================================================
// MatrixFreeGpu is an object living on the CPU, but with most of its member
// data residing on the gpu. Here, we keep all the data related to a matrix-free
// evaluation.
//=============================================================================


#define MATRIX_FREE_BKSIZE_CONSTR 128

// helper object for (re)initialization of main class
template <int dim, typename Number>
class ReinitHelper {
private:
  MatrixFreeGpu<dim,Number> *data;

  // host arrays
  std::vector<unsigned int> loc2glob_host;
  std::vector<Point<dim> > quad_points_host;
  std::vector<Number> JxW_host;
  std::vector<Number> inv_jac_host;

  std::vector<unsigned int> constraint_mask_host;

  // local buffers
  std::vector<types::global_dof_index> local_dof_indices;

  FEValues<dim> fe_values;
  // get the translation from default dof numbering to a lexicographic one
  const std::vector<unsigned int> &lexicographic_inv;
  std::vector<unsigned int> lexicographic_dof_indices;

  const unsigned int fe_degree;
  const unsigned int dofs_per_cell;
  const unsigned int qpts_per_cell;

  // TODO: fix update flags
  const UpdateFlags &update_flags;

  // For setting up hanging node constraints
#ifdef MATRIX_FREE_HANGING_NODES
  HangingNodes<dim> hanging_nodes;
#endif

  // for padding
  const unsigned int rowlength;

public:
  ReinitHelper(MatrixFreeGpu<dim,Number>                              *data,
               const Mapping<dim>                                     &mapping,
               const FiniteElement<dim>                               &fe,
               const Quadrature<1>                                    &quad,
               const internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
               const DoFHandler<dim>                                  &dof_handler,
               const UpdateFlags &update_flags)
    : data(data),
      fe_degree(data->fe_degree),
      dofs_per_cell(data->dofs_per_cell),
      qpts_per_cell(data->qpts_per_cell),
      fe_values (mapping, fe, Quadrature<dim>(quad),
                 update_inverse_jacobians | update_quadrature_points |
                 update_values | update_gradients | update_JxW_values),
      lexicographic_inv(shape_info.lexicographic_numbering),
#ifdef MATRIX_FREE_HANGING_NODES
      hanging_nodes(fe_degree,dof_handler,lexicographic_inv),
#endif
      update_flags(update_flags),
      rowlength(data->get_rowlength())
  {
    local_dof_indices.resize(data->dofs_per_cell);
    lexicographic_dof_indices.resize(dofs_per_cell);
  }

  template <typename Iterator>
  void init(const Iterator     &begin,
            const Iterator     &end,
            const dealii::ConstraintMatrix    &constraints);


  template <typename Iterator>
  void init_with_coloring(const Iterator     &begin,
                          const Iterator     &end,
                          const dealii::ConstraintMatrix    &constraints);


  void setup_color_arrays(const unsigned int num_colors);
  void setup_cell_arrays(const unsigned int c);

  /**
   * Loop over all cells from begin to end and set up data structures
   */
  template <typename Iterator>
  void cell_loop(const Iterator& begin, const Iterator& end);

  /**
   * Version used with coloring. In this case we want to loop over the resulting
   * std::vector from the coloring algorithm
   */
  template <typename CellFilter>
  void cell_loop(const typename std::vector<CellFilter>::iterator & begin,
                 const typename std::vector<CellFilter>::iterator & end);

  /**
   * Called internally from cell_loop to fill in data for one cell
   */
  template <typename T>
  void get_cell_data(const T& cell,const unsigned int cellid);

  void alloc_and_copy_arrays(const unsigned int c);

};

template <int dim, typename Number>
template <typename Iterator>
void ReinitHelper<dim,Number>::init(const Iterator     &begin,
                                    const Iterator     &end,
                                    const dealii::ConstraintMatrix    &constraints)
{
  data->num_colors = 1;

  setup_color_arrays(1);

  data->n_cells[0] = data->n_cells_tot;

  setup_cell_arrays(0);

  cell_loop(begin,end);

  // now allocate and copy stuff to the device
  alloc_and_copy_arrays(0);
}

template <int dim, typename Number>
template <typename Iterator>
void ReinitHelper<dim,Number>::init_with_coloring(const Iterator     &begin,
                                                  const Iterator     &end,
                                                  const dealii::ConstraintMatrix    &constraints)
{
  // create graph coloring

  typedef FilteredIterator<Iterator> CellFilter;
  std::vector<std::vector<CellFilter > > graph =
    GraphColoringWrapper<dim,Iterator>::make_graph_coloring(begin, end,
                                                            constraints);

  data->num_colors = graph.size();

  setup_color_arrays(data->num_colors);

  for(int c = 0; c < data->num_colors; ++c) {
    data->n_cells[c] = graph[c].size();

    setup_cell_arrays(c);

    cell_loop<CellFilter>(graph[c].begin(),
                          graph[c].end());

    // now allocate and copy stuff to the device
    alloc_and_copy_arrays(c);
  }

}

template <int dim, typename Number>
void ReinitHelper<dim,Number>::setup_color_arrays(const unsigned int num_colors)
{
  data->n_cells.resize(num_colors);
  data->grid_dim.resize(num_colors);
  data->block_dim.resize(num_colors);
  data->loc2glob.resize(num_colors);
  data->constraint_mask.resize(num_colors);

  data->rowstart.resize(num_colors);

  if(update_flags & update_quadrature_points)
    data->quadrature_points.resize(num_colors);

  if(update_flags & update_JxW_values)
    data->JxW.resize(num_colors);

  if(update_flags & update_gradients)
    data->inv_jac.resize(num_colors);
}

template <int dim, typename Number>
void ReinitHelper<dim,Number>::setup_cell_arrays(const unsigned int c)
{
  const unsigned int n_cells = data->n_cells[c];
  const unsigned int cells_per_block = data->cells_per_block;

  // setup kernel parameters
  const unsigned int apply_num_blocks = ceil(n_cells / float(cells_per_block));
  const unsigned int apply_x_num_blocks = round(sqrt(apply_num_blocks)); // get closest to even square.
  const unsigned int apply_y_num_blocks = ceil(double(apply_num_blocks)/apply_x_num_blocks);

  data->grid_dim[c] = dim3(apply_x_num_blocks,apply_y_num_blocks);

  const unsigned int n_dofs_1d = fe_degree+1;

  if(data->parallelization_scheme == MatrixFreeGpu<dim,Number>::scheme_par_in_elem) {

    if(dim==1)
      data->block_dim[c] = dim3(n_dofs_1d*cells_per_block);
    else if(dim==2)
      data->block_dim[c] = dim3(n_dofs_1d*cells_per_block,n_dofs_1d);
    else if(dim==3)
      data->block_dim[c] = dim3(n_dofs_1d*cells_per_block,n_dofs_1d,n_dofs_1d);
  }
  else {
    data->block_dim[c] = dim3(cells_per_block);
  }


  loc2glob_host.resize(n_cells*rowlength);

  if(update_flags & update_quadrature_points)
    quad_points_host.resize(n_cells*rowlength);

  if(update_flags & update_JxW_values)
    JxW_host.resize(n_cells*rowlength);

  if(update_flags & update_gradients)
    inv_jac_host.resize(n_cells*rowlength*dim*dim);

  constraint_mask_host.resize(n_cells);

}

template <int dim, typename Number>
template <typename Iterator>
void ReinitHelper<dim,Number>::cell_loop(const Iterator& begin, const Iterator& end)
{
  Iterator cell=begin;
  unsigned int cellid=0;
  for (; cell!=end; ++cell,++cellid)
    get_cell_data(cell,cellid);
}

template <int dim, typename Number>
template <typename CellFilter>
void ReinitHelper<dim,Number>::cell_loop(const typename std::vector<CellFilter>::iterator & begin,
                                         const typename std::vector<CellFilter>::iterator & end)
{
  typename std::vector<CellFilter>::iterator cell=begin;
  unsigned int cellid=0;
  for (; cell!=end; ++cell,++cellid)
    get_cell_data(*cell,cellid); // dereference iterator to get underlying cell_iterator
}

template <int dim, typename Number>
template <typename T>
void ReinitHelper<dim,Number>::get_cell_data(const T& cell, const unsigned int cellid)
{
  cell->get_active_or_mg_dof_indices(local_dof_indices);

  for(int i = 0; i < dofs_per_cell; ++i)
    lexicographic_dof_indices[i] = local_dof_indices[lexicographic_inv[i]];

  // setup hanging nodes
#ifdef MATRIX_FREE_HANGING_NODES
  hanging_nodes.setup_constraints (constraint_mask_host[cellid],
                                   lexicographic_dof_indices,
                                   cell,cellid);
#endif

  memcpy(&loc2glob_host[cellid*rowlength],&lexicographic_dof_indices[0],dofs_per_cell*sizeof(unsigned int));

  fe_values.reinit(cell);

  // quadrature points
  if(update_flags & update_quadrature_points) {
    const std::vector<Point<dim> > & qpts = fe_values.get_quadrature_points();
    memcpy(&quad_points_host[cellid*rowlength],&qpts[0],qpts_per_cell*sizeof(Point<dim>));
  }

  if(update_flags & update_JxW_values) {
    const std::vector<double > & jxws_d = fe_values.get_JxW_values();
    const unsigned int n = jxws_d.size();
    std::vector<Number > jxws(n);
    for(int i=0; i<n; ++i)
      jxws[i] = Number(jxws_d[i]);
    memcpy(&JxW_host[cellid*rowlength],&jxws[0],qpts_per_cell*sizeof(Number));
  }

  if(update_flags & update_gradients) {
    const std::vector<DerivativeForm<1,dim,dim> >& jacs = fe_values.get_inverse_jacobians();
    memcpy(&inv_jac_host[cellid*rowlength*dim*dim],&jacs[0],qpts_per_cell*sizeof(DerivativeForm<1,dim,dim>));
  }
}



template <typename T>
void transpose(T *dst, const T *src, const unsigned int N, const unsigned int M)
{
  // src is N X M
  // dst is M X N

  for(int i = 0; i < N; ++i)
    for(int j = 0; j < M; ++j)
      dst[j*N+i] = src[i*M+j];
}

// TODO: if a unified gpuarray / point would exist, only need one template argument
template <typename T>
void transpose_inplace(std::vector<T> &a_host,
                       const unsigned int n, const unsigned int m)
{
  // convert to structure-of-array
  std::vector<T> old(a_host.size());
  old.swap(a_host);

  transpose(&a_host[0],&old[0],n,m);
}


template <typename T1, typename T2>
void alloc_and_copy(T1 **a_dev, std::vector<T2> &a_host,
                    const unsigned int n)
{
  CUDA_CHECK_SUCCESS(cudaMalloc(a_dev,n*sizeof(T1)));
  CUDA_CHECK_SUCCESS(cudaMemcpy(*a_dev, &a_host[0], n*sizeof(T1),
                                cudaMemcpyHostToDevice));
}

template <int dim, typename Number>
void ReinitHelper<dim,Number>::alloc_and_copy_arrays(const unsigned int c)
{
  const unsigned n_cells = data->n_cells[c];

  // local-to-global mapping
  if(data->parallelization_scheme == MatrixFreeGpu<dim,Number>::scheme_par_over_elems) {
    transpose_inplace(loc2glob_host,n_cells, rowlength);
  }
  alloc_and_copy(&data->loc2glob[c],
                 loc2glob_host,
                 n_cells*rowlength);

  // quadrature points
  if(update_flags & update_quadrature_points) {
    if(data->parallelization_scheme == MatrixFreeGpu<dim,Number>::scheme_par_over_elems) {
      transpose_inplace(quad_points_host,n_cells, rowlength);
    }
    alloc_and_copy(&data->quadrature_points[c],
                   quad_points_host,
                   n_cells*rowlength);
  }

  // jacobian determinants/quadrature weights
  if(update_flags & update_JxW_values) {
    if(data->parallelization_scheme == MatrixFreeGpu<dim,Number>::scheme_par_over_elems) {
      transpose_inplace(JxW_host,n_cells, rowlength);
    }
    alloc_and_copy(&data->JxW[c],
                   JxW_host,
                   n_cells*rowlength);
  }
  // inverse jacobians
  if(update_flags & update_gradients) {

    // now this has index order:  cellid*qpts_per_cell*dim*dim + q*dim*dim + i
    // this is not good at all?

    // convert so that all J_11 elements are together, all J_12 elements together, etc.
    // i.e. this index order: i*qpts_per_cell*n_cells + cellid*qpts_per_cell + q
    // this is good for a dof-level parallelization

    transpose_inplace(inv_jac_host,rowlength*n_cells,dim*dim);

    // transpose second time means we get the following index order:
    // q*n_cells*dim*dim + i*n_cells + cellid
    // which is good for an element-level parallelization

    if(data->parallelization_scheme == MatrixFreeGpu<dim,Number>::scheme_par_over_elems) {
      transpose_inplace(inv_jac_host,n_cells*dim*dim, rowlength);
    }
    alloc_and_copy(&data->inv_jac[c], inv_jac_host,
                   n_cells*dim*dim*rowlength);
  }

  alloc_and_copy(&data->constraint_mask[c],constraint_mask_host,n_cells);
}


//=============================================================================
// Initialization function
//=============================================================================

template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::
reinit(const Mapping<dim>        &mapping,
       const DoFHandler<dim>     &dof_handler,
       const ConstraintMatrix &constraints,
       const Quadrature<1>           &quad,
       const AdditionalData    additional_data)
{

  if(typeid(Number) == typeid(double)) {
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }

  use_coloring = additional_data.use_coloring;

  const UpdateFlags &update_flags = additional_data.mapping_update_flags;

  if(additional_data.parallelization_scheme != scheme_par_over_elems &&
     additional_data.parallelization_scheme != scheme_par_in_elem) {
    fprintf(stderr,"Invalid parallelization scheme!\n");
    exit(1);
  }

  this->parallelization_scheme = additional_data.parallelization_scheme;

  free(); // todo, only free if we actually need arrays of different length

  const FiniteElement<dim> &fe = dof_handler.get_fe();

  fe_degree = fe.degree;
  const unsigned int n_dofs_1d = fe_degree+1;
  const unsigned int n_q_points_1d = quad.size();


  // set row length to the closest power of two larger than or equal to the number of threads
  rowlength = 1 << static_cast<unsigned int>(ceil(dim*log2(fe_degree+1.0)));

  Assert(n_dofs_1d == n_q_points_1d,ExcMessage("n_q_points_1d must be equal to fe_degree+1."));

  level_mg_handler = additional_data.level_mg_handler;

  if(level_mg_handler != numbers::invalid_unsigned_int) {
    n_dofs = dof_handler.n_dofs(level_mg_handler);
    n_cells_tot = dof_handler.get_triangulation().n_cells(level_mg_handler);
  }
  else {
    n_dofs = dof_handler.n_dofs();
    n_cells_tot = dof_handler.get_triangulation().n_active_cells();
  }

  dofs_per_cell = fe.dofs_per_cell;
  qpts_per_cell = ipowf(n_q_points_1d,dim);

  // shape info, a single copy
  const internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info(quad,fe);

  unsigned int size_shape_values = n_dofs_1d*n_q_points_1d*sizeof(Number);
  // test if  shape_info.shape_values_number.size() == (fe_degree+1)*num_quad_1d

  CUDA_CHECK_SUCCESS(cudaMemcpyToSymbol(shape_values, &shape_info.shape_values_number[0],size_shape_values));

  if(update_flags & update_gradients) {
    CUDA_CHECK_SUCCESS(cudaMemcpyToSymbol(shape_gradient, &shape_info.shape_gradient_number[0],size_shape_values));
  }

  // Setup number of cells per CUDA thread block
  cells_per_block = cells_per_block_shmem(dim,fe_degree);

  //---------------------------------------------------------------------------
  // cell-specific stuff (indices, JxW, inverse jacobian, quadrature points, etc)
  //---------------------------------------------------------------------------

  ReinitHelper<dim,Number> helper(this,mapping,fe,quad,shape_info,
                                  dof_handler,update_flags);

  if(use_coloring) {

    if(level_mg_handler != numbers::invalid_unsigned_int) {

      helper.init_with_coloring(dof_handler.begin_mg(level_mg_handler),
                                dof_handler.end_mg(level_mg_handler),
                                constraints);
    }
    else {
      const typename DoFHandler<dim>::active_cell_iterator
        begin = dof_handler.begin_active(),
        end = dof_handler.end(); // explicitly make end() an active iterator
      helper.init_with_coloring(begin,end,
                                constraints);
    }

  }
  else { // no coloring

    if(level_mg_handler != numbers::invalid_unsigned_int)
      helper.init(dof_handler.begin_mg(level_mg_handler),
                  dof_handler.end_mg(level_mg_handler),
                  constraints);
    else {
      const typename DoFHandler<dim>::active_cell_iterator
        begin = dof_handler.begin_active(),
        end = dof_handler.end(); // explicitly make end() an active iterator
      helper.init(begin,end,
                  constraints);
    }
  }

  // setup row starts
  rowstart[0] = 0;
  for(int c = 0; c < num_colors-1; ++c) {
    rowstart[c+1] = rowstart[c] +  n_cells[c] * get_rowlength();
  }

  //---------------------------------------------------------------------------
  // constrained indices
  //---------------------------------------------------------------------------

  n_constrained_dofs = constraints.n_constraints();

  const unsigned int constr_num_blocks = ceil(n_constrained_dofs / float(MATRIX_FREE_BKSIZE_CONSTR));
  const unsigned int constr_x_num_blocks = round(sqrt(constr_num_blocks)); // get closest to even square.
  const unsigned int constr_y_num_blocks = ceil(double(constr_num_blocks)/constr_x_num_blocks);

  constr_grid_dim = dim3(constr_x_num_blocks,constr_y_num_blocks);
  constr_block_dim = dim3(MATRIX_FREE_BKSIZE_CONSTR);

  std::vector<unsigned int> constrained_dofs_host(n_constrained_dofs);

  unsigned int iconstr = 0;
  for(unsigned int i=0; i<n_dofs; i++) {
    if(constraints.is_constrained(i)) {
      constrained_dofs_host[iconstr] = i;
      iconstr++;
    }
  }

  CUDA_CHECK_SUCCESS(cudaMalloc(&constrained_dofs,n_constrained_dofs*sizeof(unsigned int)));
  CUDA_CHECK_SUCCESS(cudaMemcpy(constrained_dofs, &constrained_dofs_host[0],n_constrained_dofs*sizeof(unsigned int),
                                cudaMemcpyHostToDevice));
}


template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::free()
{

  for(int c = 0; c < quadrature_points.size(); ++c) {
    if(quadrature_points[c] != NULL) CUDA_CHECK_SUCCESS(cudaFree(quadrature_points[c]));
  }
  for(int c = 0; c < loc2glob.size(); ++c) {
    if(loc2glob[c] != NULL)          CUDA_CHECK_SUCCESS(cudaFree(loc2glob[c]));
  }
  for(int c = 0; c < inv_jac.size(); ++c) {
    if(inv_jac[c] != NULL)           CUDA_CHECK_SUCCESS(cudaFree(inv_jac[c]));
  }
  for(int c = 0; c < JxW.size(); ++c) {
    if(JxW[c] != NULL)               CUDA_CHECK_SUCCESS(cudaFree(JxW[c]));
  }
  for(int c = 0; c < constraint_mask.size(); ++c) {
    if(constraint_mask[c] != NULL)   CUDA_CHECK_SUCCESS(cudaFree(constraint_mask[c]));
  }

  quadrature_points.clear();
  loc2glob.clear();
  inv_jac.clear();
  JxW.clear();
  constraint_mask.clear();


  if(constrained_dofs != NULL)  CUDA_CHECK_SUCCESS(cudaFree(constrained_dofs));
  constrained_dofs = NULL;
}


//=============================================================================
// functions dealing with constraints
//=============================================================================

template <typename Number>
__global__ void copy_constrained_dofs_kernel (Number              *dst,
                                              const Number        *src,
                                              const unsigned int  *constrained_dofs,
                                              const unsigned int  n_constrained_dofs)
{
  const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  if(dof < n_constrained_dofs) {
    dst[constrained_dofs[dof]] = src[constrained_dofs[dof]];
  }
}

template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::copy_constrained_values(GpuVector <Number> &dst,
                                                        const GpuVector<Number> &src) const
{
  if(n_constrained_dofs != 0) {

    copy_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(dst.getData(),src.getDataRO(),
                                                                                constrained_dofs,
                                                                                n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}


template <typename Number>
__global__ void set_constrained_dofs_kernel (Number               *dst,
                                             Number               val,
                                             const unsigned int   *constrained_dofs,
                                             const unsigned int   n_constrained_dofs)
{
  const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  if(dof < n_constrained_dofs) {
    dst[constrained_dofs[dof]] = val;
  }
}


template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::set_constrained_values(GpuVector <Number> &dst,
                                                       Number val) const
{
  if(n_constrained_dofs != 0) {
    set_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(dst.getData(),
                                                                               val,constrained_dofs,
                                                                               n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}

template <typename Number>
__global__ void save_constrained_dofs_kernel (const Number        *out,
                                              Number              *in,
                                              Number              *tmp_out,
                                              Number              *tmp_in,
                                              const unsigned int  *constrained_dofs,
                                              const unsigned int   n_constrained_dofs)
{
  const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  if(dof < n_constrained_dofs) {
    tmp_out[dof] = out[constrained_dofs[dof]];
    tmp_in[dof]  = in[constrained_dofs[dof]];
    in[constrained_dofs[dof]] = 0;
  }
}

template <typename Number>
__global__ void load_constrained_dofs_kernel (Number              *out,
                                              Number              *in,
                                              const Number        *tmp_out,
                                              const Number        *tmp_in,
                                              const unsigned int  *constrained_dofs,
                                              const unsigned int   n_constrained_dofs)
{
  const unsigned int dof = threadIdx.x + blockDim.x*(blockIdx.x+gridDim.x*blockIdx.y);
  if(dof < n_constrained_dofs) {
    out[constrained_dofs[dof]] = tmp_out[dof] + tmp_in[dof];
    in[constrained_dofs[dof]]  = tmp_in[dof];
  }
}


template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::save_constrained_values(const GpuVector <Number> &output,
                                                        GpuVector<Number>        &input,
                                                        GpuVector <Number>       &tmp_output,
                                                        GpuVector<Number>        &tmp_input) const
{
  if(n_constrained_dofs != 0) {
    save_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(output.getDataRO(),
                                                                                input.getData(),
                                                                                tmp_output.getData(),
                                                                                tmp_input.getData(),
                                                                                constrained_dofs,
                                                                                n_constrained_dofs);
    CUDA_CHECK_LAST;
  }

}


template <int dim, typename Number>
void MatrixFreeGpu<dim,Number>::load_constrained_values(GpuVector <Number>       &output,
                                                        GpuVector<Number>        &input,
                                                        const GpuVector <Number> &tmp_output,
                                                        const GpuVector<Number>  &tmp_input) const
{
  if(n_constrained_dofs != 0) {
    load_constrained_dofs_kernel<Number> <<<constr_grid_dim,constr_block_dim>>>(output.getData(),
                                                                                input.getData(),
                                                                                tmp_output.getDataRO(),
                                                                                tmp_input.getDataRO(),
                                                                                constrained_dofs,
                                                                                n_constrained_dofs);
    CUDA_CHECK_LAST;
  }
}
