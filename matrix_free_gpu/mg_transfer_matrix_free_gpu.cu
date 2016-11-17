// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

// #include <deal.II/lac/gpu_vector.h>
#include "cuda_utils.cuh"
#include "atomic.cuh"
#include "gpu_vec.h"
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/multigrid/mg_tools.h>
// #include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include "mg_transfer_matrix_free_gpu.h"

#include <deal.II/matrix_free/shape_info.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <algorithm>

DEAL_II_NAMESPACE_OPEN


namespace internal {

  template <typename T>
  GpuList<T>::GpuList()
  {
    n = 0;
    values = NULL;
  }

  template <typename T>
  GpuList<T>::GpuList(const GpuList<T> &other)
  {
    n = other.size();
    cudaMalloc(&values,n*sizeof(T));
    cudaAssertNoError();
    cudaMemcpy(values,other.values,n*sizeof(T),
               cudaMemcpyDeviceToDevice);
    cudaAssertNoError();
  }

  template <typename T>
  GpuList<T>::GpuList(const std::vector<T> &host_arr)
  {
    n = host_arr.size();
    cudaMalloc(&values,n*sizeof(T));
    cudaAssertNoError();
    cudaMemcpy(values,&host_arr[0],n*sizeof(T),
               cudaMemcpyHostToDevice);
    cudaAssertNoError();
  }

  template <typename T>
  GpuList<T>::~GpuList()
  {
    if(values != NULL) {
      cudaFree(values);
      cudaAssertNoError();
    }
  }

  template <typename T>
  void GpuList<T>::resize(unsigned int newsize)
  {
    if(n != newsize)
    {
      if(values != NULL) {
        cudaFree(values);
        cudaAssertNoError();
      }

      n = newsize;
      cudaMalloc(&values,n*sizeof(T));
      cudaAssertNoError();
    }
  }

  template <typename T>
  GpuList<T>& GpuList<T>::operator=(const GpuList<T> &other)
  {
    resize(other.size());

    cudaMemcpy(values,other.values,n*sizeof(T),
               cudaMemcpyDeviceToDevice);
    cudaAssertNoError();

    return *this;
  }

  template <typename T>
  GpuList<T>& GpuList<T>::operator=(const std::vector<T> &host_arr)
  {

    resize(host_arr.size());

    cudaMemcpy(values,host_arr.data(),n*sizeof(T),
               cudaMemcpyHostToDevice);
    cudaAssertNoError();

    return *this;
  }

  template <typename T>
  void GpuList<T>::clear()
  {
    n = 0;
    if(values != NULL) {
      cudaFree(values);
      cudaAssertNoError();
      values = NULL;
    }
  }

  template <typename T>
  unsigned int GpuList<T>::size() const
  {
    return n;
  }

  template <typename T>
  const T* GpuList<T>::getDataRO() const
  {
    return values;
  }

  template <typename T>
  std::size_t GpuList<T>::memory_consumption() const
  {
    return sizeof(T)*n;
  }


  std::size_t IndexMapping::memory_consumption() const
  {
    return global_indices.memory_consumption() +
      level_indices.memory_consumption();
  }

}



template<int dim, typename Number>
MGTransferMatrixFreeGpu<dim,Number>::MGTransferMatrixFreeGpu ()
  :
  fe_degree(0),
  element_is_continuous(true),
  n_components(0),
  n_child_cell_dofs(0)
{}



template<int dim, typename Number>
MGTransferMatrixFreeGpu<dim,Number>::MGTransferMatrixFreeGpu (const MGConstrainedDoFs &mg_c)
  :
  fe_degree(0),
  element_is_continuous(false),
  n_components(0),
  n_child_cell_dofs(0)
{
  this->mg_constrained_dofs = &mg_c;
}



template <int dim, typename Number>
MGTransferMatrixFreeGpu<dim,Number>::~MGTransferMatrixFreeGpu ()
{}



template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>::initialize_constraints
(const MGConstrainedDoFs &mg_c)
{
  this->mg_constrained_dofs = &mg_c;
}



template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>::clear ()
{
  // this->MGLevelGlobalTransfer<GpuVector<Number> >::clear();
  fe_degree = 0;
  element_is_continuous = false;
  n_components = 0;
  n_child_cell_dofs = 0;
  level_dof_indices.clear();
  // parent_child_connect.clear();
  child_offset_in_parent.clear();
  n_owned_level_cells.clear();
  weights_on_refined.clear();
}



namespace
{
  // given the collection of child cells in lexicographic ordering as seen
  // from the parent, compute the first index of the given child
  template <int dim>
  unsigned int
  compute_shift_within_children(const unsigned int child,
                                const unsigned int fe_shift_1d,
                                const unsigned int fe_degree)
  {
    // we put the degrees of freedom of all child cells in
    // lexicographic ordering
    unsigned int c_tensor_index[dim];
    unsigned int tmp = child;
    for (unsigned int d=0; d<dim; ++d)
    {
      c_tensor_index[d] = tmp % 2;
      tmp /= 2;
    }
    const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
    unsigned int factor = 1;
    unsigned int shift = fe_shift_1d * c_tensor_index[0];
    for (unsigned int d=1; d<dim; ++d)
    {
      factor *= n_child_dofs_1d;
      shift = shift + factor * fe_shift_1d * c_tensor_index[d];
    }
    return shift;
  }



  // puts the indices on the given child cell in lexicographic ordering with
  // respect to the collection of all child cells as seen from the parent
  template <int dim>
  void add_child_indices(const unsigned int child,
                         const unsigned int fe_shift_1d,
                         const unsigned int fe_degree,
                         const std::vector<unsigned int> &lexicographic_numbering,
                         const std::vector<types::global_dof_index> &local_dof_indices,
                         types::global_dof_index *target_indices)
  {
    const unsigned int n_child_dofs_1d = fe_degree + 1 + fe_shift_1d;
    const unsigned int shift =
      compute_shift_within_children<dim>(child, fe_shift_1d, fe_degree);
    const unsigned int n_components =
      local_dof_indices.size()/Utilities::fixed_power<dim>(fe_degree+1);
    types::global_dof_index *indices = target_indices + shift;
    const unsigned int n_scalar_cell_dofs = Utilities::fixed_power<dim>(n_child_dofs_1d);
    for (unsigned int c=0, m=0; c<n_components; ++c)
      for (unsigned int k=0; k<(dim>2 ? (fe_degree+1) : 1); ++k)
        for (unsigned int j=0; j<(dim>1 ? (fe_degree+1) : 1); ++j)
          for (unsigned int i=0; i<(fe_degree+1); ++i, ++m)
          {
            const unsigned int index = c*n_scalar_cell_dofs+k*n_child_dofs_1d*
              n_child_dofs_1d+j*n_child_dofs_1d+i;
            Assert(indices[index] == numbers::invalid_dof_index ||
                   indices[index] == local_dof_indices[lexicographic_numbering[m]],
                   ExcInternalError());
            indices[index] = local_dof_indices[lexicographic_numbering[m]];
          }
  }


  template <int dim, typename Number>
  void setup_data_structures(
                             /*
                               Finite element info
                             */
                             internal::MatrixFreeFunctions::ShapeInfo<Number> &shape_info,
                             unsigned int &n_components,
                             unsigned int &fe_degree,
                             bool &element_is_continuous,
                             unsigned int &n_child_cell_dofs,
                             /*
                               DoF indices
                             */

                             std::vector<std::vector<unsigned int>> &level_dof_indices,
                             std::vector<std::vector<std::pair<unsigned int,unsigned int> > > &parent_child_connect,
                             std::vector<unsigned int> &n_owned_level_cells,
                             // std::vector<std::vector<std::vector<unsigned short> > > &dirichlet_indices,
                             const Triangulation<dim> &tria,
                             const DoFHandler<dim> &mg_dof,
                             // SmartPointer< const MGConstrainedDoFs, MGTransferMatrixFreeGpu<dim,Number> > 	mg_constrained_dofs,

                             /*
                               Weights
                             */
                             std::vector<std::vector<Number> > &weights_on_refined,
                             const unsigned int                 n_levels)
  {

    // step 1.1: create a 1D copy of the finite element from FETools where we
    // substitute the template argument
    AssertDimension(mg_dof.get_fe().n_base_elements(), 1);
    std::string fe_name = mg_dof.get_fe().base_element(0).get_name();
    {
      const std::size_t template_starts = fe_name.find_first_of('<');
      Assert (fe_name[template_starts+1] == (dim==1?'1':(dim==2?'2':'3')),
              ExcInternalError());
      fe_name[template_starts+1] = '1';
    }
    std_cxx11::shared_ptr<FiniteElement<1> > fe_1d
      (FETools::get_fe_by_name<1,1>(fe_name));
    const FiniteElement<1> &fe = *fe_1d;
    unsigned int n_child_dofs_1d = numbers::invalid_unsigned_int;


    {
      // currently, we have only FE_Q and FE_DGQ type elements implemented
      n_components = mg_dof.get_fe().element_multiplicity(0);
      AssertDimension(Utilities::fixed_power<dim>(fe.dofs_per_cell)*n_components,
                      mg_dof.get_fe().dofs_per_cell);
      AssertDimension(fe.degree, mg_dof.get_fe().degree);
      fe_degree = fe.degree;
      element_is_continuous = fe.dofs_per_vertex > 0;
      Assert(fe.dofs_per_vertex < 2, ExcNotImplemented());

      // step 1.2: get renumbering of 1D basis functions to lexicographic
      // numbers. The distinction according to fe.dofs_per_vertex is to support
      // both continuous and discontinuous bases.
      std::vector<unsigned int> renumbering(fe.dofs_per_cell);
      {
        AssertIndexRange(fe.dofs_per_vertex, 2);
        renumbering[0] = 0;
        for (unsigned int i=0; i<fe.dofs_per_line; ++i)
          renumbering[i+fe.dofs_per_vertex] =
            GeometryInfo<1>::vertices_per_cell*fe.dofs_per_vertex + i;
        if (fe.dofs_per_vertex > 0)
          renumbering[fe.dofs_per_cell-fe.dofs_per_vertex] = fe.dofs_per_vertex;
      }

      // step 1.3: create a 1D quadrature formula from the finite element that
      // collects the support points of the basis functions on the two children.
      std::vector<Point<1> > basic_support_points = fe.get_unit_support_points();
      Assert(fe.dofs_per_vertex == 0 || fe.dofs_per_vertex == 1,
             ExcNotImplemented());
      std::vector<Point<1> > points_refined(fe.dofs_per_vertex > 0 ?
                                            (2 * fe.dofs_per_cell - 1) :
                                            (2 * fe.dofs_per_cell));
      const unsigned int shift = fe.dofs_per_cell - fe.dofs_per_vertex;
      for (unsigned int c=0; c<GeometryInfo<1>::max_children_per_cell; ++c)
        for (unsigned int j=0; j<basic_support_points.size(); ++j)
          points_refined[shift*c+j][0] =
            c*0.5 + 0.5 * basic_support_points[renumbering[j]][0];

      n_child_dofs_1d = points_refined.size();
      n_child_cell_dofs = n_components*Utilities::fixed_power<dim>(n_child_dofs_1d);

      // step 1.4: evaluate the polynomials and store the data in ShapeInfo
      const Quadrature<1> quadrature(points_refined);
      shape_info.reinit(quadrature, mg_dof.get_fe(), 0);

      for (unsigned int c=0; c<GeometryInfo<1>::max_children_per_cell; ++c)
        for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
          for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
            Assert(std::abs(shape_info.shape_values[i*n_child_dofs_1d+j+c*shift][0] -
                            fe.get_prolongation_matrix(c)(renumbering[j],renumbering[i]))
                   < std::max(2.*(double)std::numeric_limits<Number>::epsilon(),1e-12),
                   ExcInternalError());


    }
    // -------------- 2. Extract and match dof indices between child and parent


    level_dof_indices.resize(n_levels);
    parent_child_connect.resize(n_levels-1);
    n_owned_level_cells.resize(n_levels-1);
    std::vector<std::vector<unsigned int> > coarse_level_indices(n_levels-1);
    for (unsigned int level=0; level<std::min(tria.n_levels(),n_levels-1); ++level)
      coarse_level_indices[level].resize(tria.n_raw_cells(level),
                                         numbers::invalid_unsigned_int);
    std::vector<types::global_dof_index> local_dof_indices(mg_dof.get_fe().dofs_per_cell);
    // dirichlet_indices.resize(n_levels-1);

    // We use the vectors stored ghosted_level_vector in the base class for
    // keeping ghosted transfer indices. To avoid keeping two very similar
    // vectors, we merge them here.
    // if (ghosted_level_vector.max_level() != n_levels-1)
      // ghosted_level_vector.resize(0, n_levels-1);

    for (unsigned int level=n_levels-1; level > 0; --level)
    {
      unsigned int counter = 0;
      std::vector<types::global_dof_index> global_level_dof_indices;
      // std::vector<types::global_dof_index> ghosted_level_dofs;
      std::vector<types::global_dof_index> global_level_dof_indices_l0;
      // std::vector<types::global_dof_index> ghosted_level_dofs_l0;

      // step 2.1: loop over the cells on the coarse side
      typename DoFHandler<dim>::cell_iterator cell, endc = mg_dof.end(level-1);
      for (cell = mg_dof.begin(level-1); cell != endc; ++cell)
      {
        // need to look into a cell if it has children and it is locally owned
        if (!cell->has_children())
          continue;

        // step 2.2: loop through children and append the dof indices to the
        // appropriate list. We need separate lists for the owned coarse
        // cell case (which will be part of restriction/prolongation between
        // level-1 and level) and the remote case (which needs to store DoF
        // indices for the operations between level and level+1).
        AssertDimension(cell->n_children(),
                        GeometryInfo<dim>::max_children_per_cell);

        std::vector<types::global_dof_index> &next_indices = global_level_dof_indices;
        const std::size_t start_index = next_indices.size();
        next_indices.resize(start_index + n_child_cell_dofs,
                            numbers::invalid_dof_index);
        for (unsigned int c=0; c<GeometryInfo<dim>::max_children_per_cell; ++c)
        {
          cell->child(c)->get_mg_dof_indices(local_dof_indices);

          // const IndexSet &owned_level_dofs = mg_dof.locally_owned_mg_dofs(level);
          // for (unsigned int i=0; i<local_dof_indices.size(); ++i)
          //   if (!owned_level_dofs.is_element(local_dof_indices[i]))
          //     ghosted_level_dofs.push_back(local_dof_indices[i]);

          add_child_indices<dim>(c, fe.dofs_per_cell - fe.dofs_per_vertex,
                                 fe_degree, shape_info.lexicographic_numbering,
                                 local_dof_indices,
                                 &next_indices[start_index]);

          // step 2.3 store the connectivity to the parent
          if (cell->child(c)->has_children() &&
              (tria.locally_owned_subdomain()==numbers::invalid_subdomain_id
               || cell->child(c)->level_subdomain_id()==tria.locally_owned_subdomain()
               )
              )
          {
            const unsigned int child_index = coarse_level_indices[level][cell->child(c)->index()];
            AssertIndexRange(child_index, parent_child_connect[level].size());
            unsigned int parent_index = counter;
            // remote cells, i.e., cells where we work on a further
            // level but are not treated on the current level, need to
            // be placed at the end of the list; however, we do not yet
            // know the exact position in the array, so shift their
            // parent index by the number of cells so we can set the
            // correct number after the end of this loop

            // if (cell_is_remote)
              // parent_index = start_index/n_child_cell_dofs + tria.n_cells(level);
            parent_child_connect[level][child_index] =
              std::make_pair(parent_index, c);
            AssertIndexRange(mg_dof.get_fe().dofs_per_cell,
                             static_cast<unsigned short>(-1));

            // set Dirichlet boundary conditions (as a list of
            // constrained DoFs) for the child
            // if (mg_constrained_dofs != 0)
              // for (unsigned int i=0; i<mg_dof.get_fe().dofs_per_cell; ++i)
                // if (mg_constrained_dofs->is_boundary_index(level,local_dof_indices[shape_info.lexicographic_numbering[i]]))
                  // dirichlet_indices[level][child_index].push_back(i);
          }
        }
        // if (!cell_is_remote)
        // {
          AssertIndexRange(static_cast<unsigned int>(cell->index()),
                           coarse_level_indices[level-1].size());
          coarse_level_indices[level-1][cell->index()] = counter++;
        // }

        // step 2.4: include indices for the coarsest cells. we still insert
        // the indices as if they were from a child in order to use the same
        // code (the coarsest level does not matter much in terms of memory,
        // so we gain in code simplicity)
        if (level == 1)// && !cell_is_remote)
        {
          cell->get_mg_dof_indices(local_dof_indices);

          const IndexSet &owned_level_dofs_l0 = mg_dof.locally_owned_mg_dofs(0);
          // for (unsigned int i=0; i<local_dof_indices.size(); ++i)
            // if (!owned_level_dofs_l0.is_element(local_dof_indices[i]))
              // ghosted_level_dofs_l0.push_back(local_dof_indices[i]);

          const std::size_t start_index = global_level_dof_indices_l0.size();
          global_level_dof_indices_l0.resize(start_index+n_child_cell_dofs,
                                             numbers::invalid_dof_index);
          add_child_indices<dim>(0, fe.dofs_per_cell - fe.dofs_per_vertex,
                                 fe.degree, shape_info.lexicographic_numbering,
                                 local_dof_indices,
                                 &global_level_dof_indices_l0[start_index]);

          // dirichlet_indices[0].push_back(std::vector<unsigned short>());
          // if (mg_constrained_dofs != 0)
          //   for (unsigned int i=0; i<mg_dof.get_fe().dofs_per_cell; ++i)
          //     if (mg_constrained_dofs->is_boundary_index(0,local_dof_indices[shape_info.lexicographic_numbering[i]]))
          //       dirichlet_indices[0].back().push_back(i);
        }
      }

      // step 2.5: store information about the current level and prepare the
      // Dirichlet indices and parent-child relationship for the next coarser
      // level
      AssertDimension(counter*n_child_cell_dofs, global_level_dof_indices.size());
      n_owned_level_cells[level-1] = counter;
      // dirichlet_indices[level-1].resize(counter);
      parent_child_connect[level-1].
        resize(counter, std::make_pair(numbers::invalid_unsigned_int,
                                       numbers::invalid_unsigned_int));

      // step 2.6: put the cells with remotely owned parent to the end of the
      // list (these are needed for the transfer from level to level+1 but not
      // for the transfer from level-1 to level).
      if (level < n_levels-1)
        for (std::vector<std::pair<unsigned int,unsigned int> >::iterator
               i=parent_child_connect[level].begin(); i!=parent_child_connect[level].end(); ++i)
          if (i->first >= tria.n_cells(level))
          {
            i->first -= tria.n_cells(level);
            i->first += counter;
          }

      // step 2.7: copy level dof indices

      level_dof_indices[level].resize(global_level_dof_indices.size(),
                               numbers::invalid_unsigned_int);

      for (unsigned int i=0; i<global_level_dof_indices.size(); ++i)
        if (global_level_dof_indices[i] != numbers::invalid_dof_index)
          level_dof_indices[level][i] = global_level_dof_indices[i];


      // step 2.8: Initialize the ghosted vector for level 0
      if (level == 1)
      {
        for (unsigned int i = 0; i<parent_child_connect[0].size(); ++i)
          parent_child_connect[0][i] = std::make_pair(i, 0U);


        level_dof_indices[0].resize(global_level_dof_indices_l0.size(),
                                    numbers::invalid_unsigned_int);

        for (unsigned int i=0; i<global_level_dof_indices_l0.size(); ++i)
          if (global_level_dof_indices_l0[i] != numbers::invalid_dof_index)
            level_dof_indices[0][i] = global_level_dof_indices_l0[i];

      }
    }


    // ------------------------ 3. compute weights to make restriction additive
    //

    std::vector<Number> level_vector;

    // get the valence of the individual components and compute the weights as
    // the inverse of the valence
    weights_on_refined.resize(n_levels-1);

    for (unsigned int level = 1; level<n_levels; ++level)
    {
      level_vector.assign(n_child_cell_dofs*n_owned_level_cells[level-1],0.0);

      for (unsigned int c=0; c<n_owned_level_cells[level-1]; ++c)
        for (unsigned int j=0; j<n_child_cell_dofs; ++j)
          level_vector[level_dof_indices[level][n_child_cell_dofs*c+j]] += Number(1.);

      std::vector<unsigned int> degree_to_3 (n_child_dofs_1d);
      degree_to_3[0] = 0;
      for (unsigned int i=1; i<n_child_dofs_1d-1; ++i)
        degree_to_3[i] = 1;
      degree_to_3.back() = 2;

      // we only store 3^dim weights because all dofs on a line have the same
      // valence, and all dofs on a quad have the same valence.
      weights_on_refined[level-1].resize(n_owned_level_cells[level-1]*Utilities::fixed_power<dim>(3));
      for (unsigned int c=0; c<n_owned_level_cells[level-1]; ++c)
      {

        for (unsigned int k=0, m=0; k<(dim>2 ? n_child_dofs_1d : 1); ++k)
          for (unsigned int j=0; j<(dim>1 ? n_child_dofs_1d : 1); ++j)
          {
            unsigned int shift = 9*degree_to_3[k] + 3*degree_to_3[j];
            for (unsigned int i=0; i<n_child_dofs_1d; ++i, ++m)
              weights_on_refined[level-1][c*Utilities::fixed_power<dim>(3)+shift+degree_to_3[i]] = Number(1.)/
                // ghosted_level_vector[level].local_element(level_dof_indices[level][n_child_cell_dofs*c+m]);
                // ghosted_level_vector[level][level_dof_indices[level][n_child_cell_dofs*c+m]];
                level_vector[level_dof_indices[level][n_child_cell_dofs*c+m]];
          }
      }
    }

  }
}


template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>::setup_copy_indices(const DoFHandler<dim,dim>  &mg_dof)
{

  std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index> > > copy_indices;
  std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index> > > copy_indices_global_mine;
  std::vector<std::vector<std::pair<types::global_dof_index, types::global_dof_index> > > copy_indices_level_mine;

  internal::fill_copy_indices<dim,dim> (mg_dof, mg_constrained_dofs, copy_indices,
                                        copy_indices_global_mine, copy_indices_level_mine);

  const unsigned int nlevels = mg_dof.get_triangulation().n_global_levels();


  Assert((copy_indices_global_mine.size() == 0) &&
         (copy_indices_level_mine.size() == 0),
         ExcMessage("Only implemented for non-distributed case"));

  for(int i=0; i<nlevels; ++i) {
    Assert((copy_indices[i].size() == copy_indices_global_mine[i].size()) &&
           (copy_indices[i].size() == copy_indices_level_mine[i].size()),
           ExcMessage("Length mismatch of copy_indices* subarrays"));
    for(int j=0; j<copy_indices[i].size(); ++j) {
      Assert((copy_indices[i][j] == copy_indices_global_mine[i][j]) &&
             (copy_indices[i][j] == copy_indices_level_mine[i][j]),
             ExcMessage("content mismatch of copy_indices* mappings"));
    }
  }

  // all set, know we can safely throw away the latter two arrays

  this->copy_indices.resize(nlevels);

  for(int i=0; i<nlevels; ++i) {
    const unsigned int nmappings = copy_indices[i].size();
    std::vector<int> global_indices(nmappings);
    std::vector<int> level_indices(nmappings);

    for(int j=0; j<nmappings; ++j) {
      global_indices[j] = copy_indices[i][j].first;
      level_indices[j] = copy_indices[i][j].second;
    }

    this->copy_indices[i].global_indices = global_indices;
    this->copy_indices[i].level_indices = level_indices;
  }


  // check if we can run a plain copy operation between the global DoFs and
  // the finest level.
  perform_plain_copy =
    (copy_indices.back().size() == mg_dof.locally_owned_dofs().n_elements())
    &&
    (mg_dof.locally_owned_dofs().n_elements() ==
     mg_dof.locally_owned_mg_dofs(mg_dof.get_triangulation().n_global_levels()-1).n_elements());

  if (perform_plain_copy)
    {
      // AssertDimension(copy_indices_global_mine.back().size(), 0);
      // AssertDimension(copy_indices_level_mine.back().size(), 0);

      // check whether there is a renumbering of degrees of freedom on
      // either the finest level or the global dofs, which means that we
      // cannot apply a plain copy
      for (unsigned int i=0; i<copy_indices.back().size(); ++i)
        if (copy_indices.back()[i].first != copy_indices.back()[i].second)
          {
            perform_plain_copy = false;
            break;
          }
    }

}


template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>::build
(const DoFHandler<dim,dim>  &mg_dof)
{
  setup_copy_indices(mg_dof);

  const Triangulation<dim> &tria = mg_dof.get_triangulation();

  std::vector<std::vector<Number>> weights_host;
  std::vector<std::vector<unsigned int>> level_dof_indices_host;
  internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
  std::vector<std::vector<std::pair<unsigned int,unsigned int> > > parent_child_connect;

  const unsigned int n_levels = tria.n_global_levels();

  // we collect all child DoFs of a mother cell together. For faster
  // tensorized operations, we align the degrees of freedom
  // lexicographically. We distinguish FE_Q elements and FE_DGQ elements

  // ---------------------------- 1. Extract 1D info about the finite element
  setup_data_structures<dim> (
                              /*
                                Finite element info
                               */
                              shape_info,                  // write
                              n_components,                // write
                              fe_degree,                   // write
                              element_is_continuous,       // write
                              // n_child_dofs_1d,             // write
                              n_child_cell_dofs,           // write

                              /*
                                DoF indices
                               */
                              level_dof_indices_host, // write
                              parent_child_connect,   // write
                              n_owned_level_cells,    // write
                              // dirichlet_indices,      // write
                              tria,                   // read ..
                              mg_dof,                 // read
                              // mg_constrained_dofs,

                              /*
                                Weights
                               */
                              weights_host, // write
                              n_levels);


  //---------------------------------------------------------------------------
  // transfer stuff from host to device
  //---------------------------------------------------------------------------

  const unsigned int size_shape_values = shape_info.shape_values_number.size();
  shape_values.resize(size_shape_values);
  shape_values.fromHost(&shape_info.shape_values_number[0],size_shape_values);

  level_dof_indices.resize(n_levels);

  for(int l=0; l<n_levels; l++) {
    level_dof_indices[l]=level_dof_indices_host[l];
  }

  weights_on_refined.resize(n_levels-1);
  for(int l=0; l<n_levels-1; l++) {
    weights_on_refined[l] = weights_host[l];
  }



  child_offset_in_parent.resize(n_levels-1);
  std::vector<unsigned int> offsets;

  for(int l=0; l<n_levels-1; l++) {

    offsets.resize(n_owned_level_cells[l]);

    for(int c=0; c<n_owned_level_cells[l]; ++c) {
      const unsigned int shift = compute_shift_within_children<dim> (parent_child_connect[l][c].second,
                                                                     fe_degree, fe_degree);
      offsets[c] = parent_child_connect[l][c].first*n_child_cell_dofs + shift;
    }

    child_offset_in_parent[l] = offsets;

  }

  std::vector<types::global_dof_index> dirichlet_index_vector;

  dirichlet_indices.resize(n_levels); // FIXME: ?

  if(mg_constrained_dofs != NULL) {

    for(int l=0; l<n_levels; l++) {

      mg_constrained_dofs->get_boundary_indices(l).fill_index_vector(dirichlet_index_vector);

      dirichlet_indices[l] = dirichlet_index_vector;
    }
  }

}


enum prol_restr {
  PROLONGATION, RESTRICTION
};


template <int dim, int fe_degree, typename Number>
class MGTransferBody
{
protected:
  static const unsigned int n_coarse = fe_degree+1;
  static const unsigned int n_fine = fe_degree*2+1;
  static const unsigned int M = 2;
  Number *values;
  const Number *weights;
  const Number *shape_values;
  const unsigned int *dof_indices_coarse;
  const unsigned int *dof_indices_fine;

  __device__ MGTransferBody(Number *buf, const Number *w,
                            const Number *shvals,
                            const unsigned int *idx_coarse,
                            const unsigned int *idx_fine)
    : values(buf), weights(w), shape_values(shvals),
      dof_indices_coarse(idx_coarse), dof_indices_fine(idx_fine) {}

  template <prol_restr red_type, int dir>
  __device__ void reduce(const Number *my_shvals)
  {
    // multiplicity of large and small size
    const bool prol = red_type==PROLONGATION;
    const unsigned int n_src = prol ? n_coarse : n_fine;

    // in direction of reduction (dir and threadIdx.x respectively), always read
    // from 1 location, and write to M (typically 2). in other directions, either
    // read M or 1 and write same number.
    const unsigned int M1 = prol?M:1;
    const unsigned int M2 = prol?(dir>0?M:1):((dir>0||dim<2)?1:M);
    const unsigned int M3 = prol?(dir>1?M:1):((dir>1||dim<3)?1:M);

    const bool last_thread_x = threadIdx.x==(n_coarse-1);
    const bool last_thread_y = threadIdx.y==(n_coarse-1);
    const bool last_thread_z = threadIdx.z==(n_coarse-1);

    Number tmp[M1*M2*M3];

#pragma unroll
    for(int m3=0; m3<M3; ++m3) {
#pragma unroll
      for(int m2=0; m2<M2 ; ++m2) {
#pragma unroll
        for(int m1=0; m1<M1; ++m1) {

          tmp[m1+M1*(m2 + M2*m3)] = 0;

          for(int i=0; i<n_src; ++i) {
            const unsigned int x=i;
            const unsigned int y=m2+M2*threadIdx.y;
            const unsigned int z=m3+M3*threadIdx.z;
            const unsigned int idx = (dir==0 ? x +n_fine*(y + n_fine*z)
                                      : dir==1 ? y + n_fine*(x + n_fine*z)
                                      :         y +n_fine*(z + n_fine*x));
            // unless we are the last thread in a direction AND we are updating
            // any value after the first one, go ahead
            if(((m1==0) || !last_thread_x) &&
               ((m2==0) || !last_thread_y) &&
               ((m3==0) || !last_thread_z)) {
              tmp[m1+M1*(m2 + M2*m3)] += my_shvals[m1*n_src+i]*values[idx];
            }
          }
        }
      }
    }
    __syncthreads();

#pragma unroll
    for(int m3=0; m3<M3; ++m3) {
#pragma unroll
      for(int m2=0; m2<M2; ++m2) {
#pragma unroll
        for(int m1=0; m1<M1; ++m1) {
          const unsigned int x=m1+M1*threadIdx.x;
          const unsigned int y=m2+M2*threadIdx.y;
          const unsigned int z=m3+M3*threadIdx.z;
          const unsigned int idx = (dir==0 ? x +n_fine*(y + n_fine*z)
                                    : dir==1 ? y + n_fine*(x + n_fine*z)
                                    :         y +n_fine*(z + n_fine*x));

          if(((m1==0) || !last_thread_x) &&
             ((m2==0) || !last_thread_y) &&
             ((m3==0) || !last_thread_z)) {
            values[idx] = tmp[m1 + M1*(m2 + M2*m3)];
          }
        }
      }
    }
  }

  inline __device__ unsigned int dof1d_to_3(unsigned int x)
  {
    return (x>0) + (x==(fe_degree*2));
  }

  __device__ void weigh_values()
  {
    const unsigned int M1 = M;
    const unsigned int M2 = (dim>1?M:1);
    const unsigned int M3 = (dim>2?M:1);

#pragma unroll
    for(int m3=0; m3<M3; ++m3) {
#pragma unroll
      for(int m2=0; m2<M2; ++m2) {
#pragma unroll
        for(int m1=0; m1<M1; ++m1) {
          const unsigned int x = (M1*threadIdx.x+m1);
          const unsigned int y = (M2*threadIdx.y+m2);
          const unsigned int z = (M3*threadIdx.z+m3);

          const unsigned int idx = x + n_fine*(y + n_fine*z);
          const unsigned int weight_idx = dof1d_to_3(x) +3*(dof1d_to_3(y) + 3*dof1d_to_3(z));

          if(x<n_fine && y<n_fine && z<n_fine) {
            values[idx] *= weights[weight_idx];
          }
        }
      }
    }
  }
};


template <int dim, int fe_degree, typename Number>
class MGProlongateBody : public MGTransferBody<dim,fe_degree,Number>
{
  using MGTransferBody<dim,fe_degree,Number>::M;
  using MGTransferBody<dim,fe_degree,Number>::n_coarse;
  using MGTransferBody<dim,fe_degree,Number>::n_fine;
  using MGTransferBody<dim,fe_degree,Number>::dof_indices_coarse;
  using MGTransferBody<dim,fe_degree,Number>::dof_indices_fine;
  using MGTransferBody<dim,fe_degree,Number>::values;
  using MGTransferBody<dim,fe_degree,Number>::shape_values;
  using MGTransferBody<dim,fe_degree,Number>::weights;
private:
  __device__ void read_coarse(const Number *vec)
  {
    const unsigned int idx = threadIdx.x + n_fine*(threadIdx.y + n_fine*threadIdx.z);
    values[idx] = vec[dof_indices_coarse[idx]];
  }

  __device__ void write_fine(Number *vec) const
  {
    const unsigned int M1 = M;
    const unsigned int M2 = (dim>1?M:1);
    const unsigned int M3 = (dim>2?M:1);

    for(int m3=0; m3<M3; ++m3)
      for(int m2=0; m2<M2; ++m2)
        for(int m1=0; m1<M1; ++m1) {
          const unsigned int x = (M1*threadIdx.x+m1);
          const unsigned int y = (M2*threadIdx.y+m2);
          const unsigned int z = (M3*threadIdx.z+m3);

          const unsigned int idx = x + n_fine*(y + n_fine*z);
          if(x<n_fine && y<n_fine && z<n_fine)
            atomicAddWrapper(&vec[dof_indices_fine[idx]],values[idx]);
        }
  }

public:
  __device__ MGProlongateBody(Number *buf, const Number *w,
                              const Number *shvals,
                              const unsigned int *idx_coarse,
                              const unsigned int *idx_fine)
    : MGTransferBody<dim,fe_degree,Number>(buf, w, shvals,
                                           idx_coarse, idx_fine) {}

  __device__ void run(Number *dst, const Number *src)
  {

    Number my_shvals[M*n_coarse];
    for(int m=0; m<(threadIdx.x<fe_degree?M:1); ++m)
      for(int i=0; i<n_coarse; ++i)
        my_shvals[m*n_coarse + i] = shape_values[(threadIdx.x*M+m) + n_fine*i];

    read_coarse(src);
    __syncthreads();

    this->template reduce<PROLONGATION,0>(my_shvals);
    __syncthreads();
    if(dim>1) {
      this->template reduce<PROLONGATION,1>(my_shvals);
      __syncthreads();
      if(dim>2) {
        this->template reduce<PROLONGATION,2>(my_shvals);
        __syncthreads();
      }
    }

    this->weigh_values();
    __syncthreads();

    write_fine(dst);
  }
};


template <int dim, int fe_degree, typename Number>
class MGRestrictBody : public MGTransferBody<dim,fe_degree,Number>
{
  using MGTransferBody<dim,fe_degree,Number>::M;
  using MGTransferBody<dim,fe_degree,Number>::n_coarse;
  using MGTransferBody<dim,fe_degree,Number>::n_fine;
  using MGTransferBody<dim,fe_degree,Number>::dof_indices_coarse;
  using MGTransferBody<dim,fe_degree,Number>::dof_indices_fine;
  using MGTransferBody<dim,fe_degree,Number>::values;
  using MGTransferBody<dim,fe_degree,Number>::shape_values;
  using MGTransferBody<dim,fe_degree,Number>::weights;
private:
  __device__ void read_fine(const Number *vec)
  {
    const unsigned int M1 = M;
    const unsigned int M2 = (dim>1?M:1);
    const unsigned int M3 = (dim>2?M:1);

    for(int m3=0; m3<M3; ++m3)
      for(int m2=0; m2<M2; ++m2)
        for(int m1=0; m1<M1; ++m1) {
          const unsigned int x = (M1*threadIdx.x+m1);
          const unsigned int y = (M2*threadIdx.y+m2);
          const unsigned int z = (M3*threadIdx.z+m3);

          const unsigned int idx = x + n_fine*(y + n_fine*z);
          if(x<n_fine && y<n_fine && z<n_fine)
            values[idx] = vec[dof_indices_fine[idx]];
        }
  }

  __device__ void write_coarse(Number *vec) const
  {
    const unsigned int idx = threadIdx.x + n_fine*(threadIdx.y + n_fine*threadIdx.z);
    atomicAddWrapper(&vec[dof_indices_coarse[idx]],values[idx]);
  }


public:

  __device__ MGRestrictBody(Number *buf, const Number *w,
                            const Number *shvals,
                            const unsigned int *idx_coarse,
                            const unsigned int *idx_fine)
    : MGTransferBody<dim,fe_degree,Number>(buf, w, shvals,
                                           idx_coarse, idx_fine) {}

  __device__ void run(Number *dst, const Number *src)
  {
    Number my_shvals[n_fine];
    for(int i=0; i<n_fine; ++i)
      my_shvals[i] = shape_values[threadIdx.x*n_fine + i];

    read_fine(src);
    __syncthreads();
    this->weigh_values();
    __syncthreads();

    this->template reduce<RESTRICTION,0>(my_shvals);
    __syncthreads();
    if(dim>1) {
      this->template reduce<RESTRICTION,1>(my_shvals);
      __syncthreads();
      if(dim>2) {
        this->template reduce<RESTRICTION,2>(my_shvals);
        __syncthreads();
      }
    }

    write_coarse(dst);
  }
};


template <int dim, int degree, typename loop_body, typename Number>
__global__ void mg_kernel (Number *dst, const Number *src, const Number *weights, const Number *shape_values,
                           const unsigned int *dof_indices_coarse, const unsigned int *dof_indices_fine,
                           const unsigned int *child_offset_in_parent,
                           const unsigned int n_child_cell_dofs)
{
  const unsigned int n_fine = Utilities::fixed_int_power<degree*2+1,dim>::value;
  const unsigned int coarse_cell = blockIdx.x;
  const unsigned int coarse_offset = child_offset_in_parent[coarse_cell];
  __shared__ Number values[n_fine];

  loop_body body(values, weights+coarse_cell*Utilities::fixed_int_power<3,dim>::value,
                 shape_values, dof_indices_coarse+coarse_offset,
                 dof_indices_fine+coarse_cell*n_child_cell_dofs);

  body.run(dst, src);
}

template <int dim, typename Number>
template <template <int,int,typename> class loop_body, int degree>
void MGTransferMatrixFreeGpu<dim,Number>
::coarse_cell_loop  (const unsigned int      fine_level,
                     GpuVector<Number>       &dst,
                     const GpuVector<Number> &src) const
{
  // const unsigned int n_fine_dofs_1d = 2*degree+1;
  const unsigned int n_coarse_dofs_1d = degree+1;

  const unsigned int n_coarse_cells = n_owned_level_cells[fine_level-1];

  // kernel parameters
  dim3 bk_dim(n_coarse_dofs_1d,
              (dim>1)?n_coarse_dofs_1d:1,
              (dim>2)?n_coarse_dofs_1d:1);

  printf("coarse cells: %d\n",n_coarse_cells);
  dim3 gd_dim(n_coarse_cells);

  mg_kernel<dim, degree, loop_body<dim,degree,Number> >
    <<<gd_dim,bk_dim>>> (dst.getData(),
                         src.getDataRO(),
                         weights_on_refined[fine_level-1].getDataRO(), // only has fine-level entries
                         shape_values.getDataRO(),
                         level_dof_indices[fine_level-1].getDataRO(),
                         level_dof_indices[fine_level].getDataRO(),
                         child_offset_in_parent[fine_level-1].getDataRO(), // on coarse level
                         n_child_cell_dofs);


  cudaAssertNoError();
}


template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>
::prolongate (const unsigned int       to_level,
              GpuVector<Number>       &dst,
              const GpuVector<Number> &src) const
{
  Assert ((to_level >= 1) && (to_level<=level_dof_indices.size()),
          ExcIndexRange (to_level, 1, level_dof_indices.size()+1));

  // AssertDimension(this->ghosted_level_vector[to_level].local_size(),
  //                 dst.local_size());
  // AssertDimension(this->ghosted_level_vector[to_level-1].local_size(),
  //                 src.local_size());

  // this->ghosted_level_vector[to_level-1] = src;
  // this->ghosted_level_vector[to_level] = 0.;
  dst = 0;

  GpuVector<Number> src_with_bc(src);
  set_constrained_dofs(src_with_bc,to_level-1,0);

  /*if (fe_degree == 0)
    coarse_cell_loop<MGProlongateBody,0>(to_level, dst, src);
  else
  */
  if (fe_degree == 1)
    coarse_cell_loop<MGProlongateBody,1>(to_level, dst, src_with_bc);
  else if (fe_degree == 2)
    coarse_cell_loop<MGProlongateBody,2>(to_level, dst, src_with_bc);
  else if (fe_degree == 3)
    coarse_cell_loop<MGProlongateBody,3>(to_level, dst, src_with_bc);
  else if (fe_degree == 4)
    coarse_cell_loop<MGProlongateBody,4>(to_level, dst, src_with_bc);
  // else if (fe_degree == 5)
    // coarse_cell_loop<MGProlongateBody,5>(to_level, dst, src);
  // else if (fe_degree == 6)
    // coarse_cell_loop<MGProlongateBody,6>(to_level, dst, src);
  // else if (fe_degree == 7)
    // coarse_cell_loop<MGProlongateBody,7>(to_level, dst, src);
  // else if (fe_degree == 8)
    // coarse_cell_loop<MGProlongateBody,8>(to_level, dst, src);
  // else if (fe_degree == 9)
    // coarse_cell_loop<MGProlongateBody,9>(to_level, dst, src);
  // else if (fe_degree == 10)
    // coarse_cell_loop<MGProlongateBody,10>(to_level, dst, src);
  else
    AssertThrow(false, ExcNotImplemented("Only degrees 0 up to 10 implemented."));


  // now set constrained dofs to 0

  // set_constrained_dofs(dst,to_level,0);

  // this->ghosted_level_vector[to_level].compress(VectorOperation::add);
  // dst = this->ghosted_level_vector[to_level];
}



template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>
::restrict_and_add (const unsigned int      from_level,
                    GpuVector<Number>       &dst,
                    const GpuVector<Number> &src) const
{
  Assert ((from_level >= 1) && (from_level<=level_dof_indices.size()),
          ExcIndexRange (from_level, 1, level_dof_indices.size()+1));

  // AssertDimension(this->ghosted_level_vector[from_level].local_size(),
  //                 src.local_size());
  // AssertDimension(this->ghosted_level_vector[from_level-1].local_size(),
  //                 dst.local_size());

  // this->ghosted_level_vector[from_level] = src;
  // this->ghosted_level_vector[from_level].update_ghost_values();
  // this->ghosted_level_vector[from_level-1] = 0.;

  GpuVector<Number> increment;
  increment.reinit(dst,false); // resize to correct size and initialize to 0

  /*if (fe_degree == 0)
    coarse_cell_loop<MGRestrictBody,0>(from_level, increment, src);
  else
  */
  if (fe_degree == 1)
    coarse_cell_loop<MGRestrictBody,1>(from_level, increment, src);
  else if (fe_degree == 2)
    coarse_cell_loop<MGRestrictBody,2>(from_level, increment, src);
  else if (fe_degree == 3)
    coarse_cell_loop<MGRestrictBody,3>(from_level, increment, src);
  else if (fe_degree == 4)
    coarse_cell_loop<MGRestrictBody,4>(from_level, increment, src);
  // else if (fe_degree == 5)
    // coarse_cell_loop<MGRestrictBody,5>(from_level, increment, src);
  // else if (fe_degree == 6)
    // coarse_cell_loop<MGRestrictBody,6>(from_level, increment, src);
  // else if (fe_degree == 7)
    // coarse_cell_loop<MGRestrictBody,7>(from_level, increment, src);
  // else if (fe_degree == 8)
    // coarse_cell_loop<MGRestrictBody,8>(from_level, increment, src);
  // else if (fe_degree == 9)
    // coarse_cell_loop<MGRestrictBody,9>(from_level, increment, src);
  // else if (fe_degree == 10)
    // coarse_cell_loop<MGRestrictBody,10>(from_level, increment, src);
  else
    AssertThrow(false, ExcNotImplemented("Only degrees 0 up to 10 implemented."));

  // this->ghosted_level_vector[from_level-1].compress(VectorOperation::add);
  // dst += this->ghosted_level_vector[from_level-1];

  set_constrained_dofs(increment,from_level-1,0);

  dst.add(increment);

}

template <typename Number>
__global__ void set_constrained_dofs_kernel(Number *vec, const unsigned int *indices,
                                            unsigned int len, Number val)
{
  const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx < len) {
    vec[indices[idx]] = val;
  }
}

template <int dim, typename Number>
void MGTransferMatrixFreeGpu<dim,Number>::set_constrained_dofs(GpuVector<Number>& vec,
                                                               unsigned int level,
                                                               Number val) const
{
  const unsigned int bksize = 256;
  const unsigned int len = dirichlet_indices[level].size();
  const unsigned int nblocks = (len-1)/bksize + 1;
  dim3 bk_dim(bksize);
  dim3 gd_dim(nblocks);

  set_constrained_dofs_kernel<<<gd_dim,bk_dim>>>(vec.getData(),
                                                 dirichlet_indices[level].getDataRO(),
                                                 len,val);
  cudaAssertNoError();
}

template <typename Number>
__global__ void copy_with_indices_kernel(Number *dst, const Number *src, const int *dst_indices, const int *src_indices, int n)
{
  const int i = threadIdx.x + blockIdx.x*blockDim.x;
  if(i<n) {
    dst[dst_indices[i]] = src[src_indices[i]];
  }
}

template <typename Number>
void copy_with_indices(GpuVector<Number> &dst, const GpuVector<Number> &src,
                       const internal::GpuList<int> &dst_indices, const internal::GpuList<int> &src_indices)
{
  const int n = dst_indices.size();
  const int blocksize = 256;
  const dim3 block_dim = dim3(blocksize);
  const dim3 grid_dim = dim3(1 + (n-1)/blocksize);
  copy_with_indices_kernel<<<grid_dim, block_dim >>>(dst.getData(),src.getDataRO(),dst_indices.getDataRO(),src_indices.getDataRO(),n);
}

template <int dim, typename Number>
template <int spacedim>
void
MGTransferMatrixFreeGpu<dim,Number>::copy_to_mg (const DoFHandler<dim,spacedim>    &mg_dof,
                                                 MGLevelObject<GpuVector<Number> > &dst,
                                                 const GpuVector<Number>           &src) const
{
  AssertIndexRange(dst.max_level(), mg_dof_handler.get_triangulation().n_global_levels());
  AssertIndexRange(dst.min_level(), dst.max_level()+1);

  for (unsigned int level=dst.min_level();
       level<=dst.max_level(); ++level)
  {
    unsigned int n = mg_dof.n_dofs (level);
    dst[level].reinit(n);
  }


#ifdef DEBUG_OUTPUT
  std::cout << "copy_to_mg src " << src.l2_norm() << std::endl;
#endif

  if (perform_plain_copy)
    {
      // if the finest multigrid level covers the whole domain (i.e., no
      // adaptive refinement) and the numbering of the finest level DoFs and
      // the global DoFs are the same, we can do a plain copy
      AssertDimension(dst[dst.max_level()].size(), src.size());

      dst[dst.max_level()] = src;

      return;
    }

  for (unsigned int level=dst.max_level(); level >= dst.min_level(); --level)
    {
      GpuVector<Number> &dst_level = dst[level];

      copy_with_indices(dst_level,src,
                        copy_indices[level].level_indices,
                        copy_indices[level].global_indices);

#ifdef DEBUG_OUTPUT
      std::cout << "copy_to_mg dst " << level << " " << dst_level.l2_norm() << std::endl;
#endif
    }
}



template <int dim, typename Number>
template <int spacedim>
void
MGTransferMatrixFreeGpu<dim,Number>::copy_from_mg (const DoFHandler<dim,spacedim>         &mg_dof,
                                                   GpuVector<Number>                      &dst,
                                                   const MGLevelObject<GpuVector<Number>> &src) const
{
  AssertIndexRange(src.max_level(), mg_dof_handler.get_triangulation().n_global_levels());
  AssertIndexRange(src.min_level(), src.max_level()+1);

  if (perform_plain_copy)
    {
      AssertDimension(dst.size(), src[src.max_level()].size());
      dst = src[src.max_level()];
      return;
    }

  dst = 0;
  for (unsigned int level=src.min_level(); level<=src.max_level(); ++level)
    {
#ifdef DEBUG_OUTPUT
      std::cout << "copy_from_mg src " << level << " " << src[level].l2_norm() << std::endl;
#endif

      const GpuVector<Number> &src_level = src[level];

      copy_with_indices(dst,src_level,
                        copy_indices[level].global_indices,
                        copy_indices[level].level_indices);

#ifdef DEBUG_OUTPUT
        std::cout << "copy_from_mg level=" << level << " " << dst.l2_norm() << std::endl;
#endif
    }
}


template <int dim, typename Number>
template <int spacedim>
void
MGTransferMatrixFreeGpu<dim,Number>::copy_from_mg_add (const DoFHandler<dim,spacedim>         &mg_dof,
                                                       GpuVector<Number>                      &dst,
                                                       const MGLevelObject<GpuVector<Number>> &src) const
{
  ExcNotImplemented();
}



template <int dim, typename Number>
std::size_t
MGTransferMatrixFreeGpu<dim,Number>::memory_consumption() const
{
  std::size_t memory = 0; // MGLevelGlobalTransfer<GpuVector<Number> >::memory_consumption();
  memory += MemoryConsumption::memory_consumption(copy_indices);
  memory += MemoryConsumption::memory_consumption(level_dof_indices);
  memory += MemoryConsumption::memory_consumption(child_offset_in_parent);
  memory += MemoryConsumption::memory_consumption(n_owned_level_cells);
  memory += shape_values.memory_consumption();
  memory += MemoryConsumption::memory_consumption(weights_on_refined);
  memory += MemoryConsumption::memory_consumption(dirichlet_indices);
  return memory;
}


//=============================================================================
// explicit instantiations
//=============================================================================

// #include "mg_transfer_matrix_free.inst"

// template class MGTransferMatrixFreeGpu<2,double>;
template class MGTransferMatrixFreeGpu<3,double>;

template void
MGTransferMatrixFreeGpu<3,double>::copy_to_mg (const DoFHandler<3>&,
                                               MGLevelObject<GpuVector<double> >&,
                                               const GpuVector<double>&) const;

template void
MGTransferMatrixFreeGpu<3,double>::copy_from_mg (const DoFHandler<3>&,
                                                 GpuVector<double>&,
                                                 const MGLevelObject<GpuVector<double> >&) const;


//=============================================================================
// FIXME: is this really the only way to do this?

template <typename VectorType>
MGTransferBase<VectorType>::~MGTransferBase()
{}


template <typename VectorType>
MGMatrixBase<VectorType>::~MGMatrixBase()
{}


template <typename VectorType>
MGSmootherBase<VectorType>::~MGSmootherBase()
{}


template <typename VectorType>
MGCoarseGridBase<VectorType>::~MGCoarseGridBase()
{}
template class MGTransferBase< GpuVector<double> >;
template class MGMatrixBase<GpuVector<double> >;
template class MGSmootherBase< GpuVector<double> >;
template class MGCoarseGridBase< GpuVector<double> >;


//=============================================================================




DEAL_II_NAMESPACE_CLOSE
