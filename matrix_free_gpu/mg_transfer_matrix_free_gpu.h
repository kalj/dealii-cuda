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

#ifndef dealii__mg_transfer_matrix_free_gpu_h
#define dealii__mg_transfer_matrix_free_gpu_h

#include <deal.II/base/config.h>

// #include <deal.II/lac/gpu_vector.h>
#include <deal.II/multigrid/mg_base.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/base/mg_level_object.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/dofs/dof_handler.h>

#include "gpu_vec.h"
#include "gpu_list.h"

DEAL_II_NAMESPACE_OPEN

namespace internal {

  struct IndexMapping {
    GpuList<int> global_indices;
    GpuList<int> level_indices;

    std::size_t memory_consumption() const;
  };

}

/*!@addtogroup mg */
/*@{*/

/**
 * Implementation of the MGTransferBase interface for which the transfer
 * operations is implemented in a matrix-free way based on the interpolation
 * matrices of the underlying finite element. This requires considerably less
 * memory than MGTransferPrebuilt and can also be considerably faster than
 * that variant.
 *
 * This class currently only works for tensor-product finite elements based on
 * FE_Q and FE_DGQ elements, including systems involving multiple components
 * of one of these elements. Systems with different elements or other elements
 * are currently not implemented.
 *
 * @author Karl Ljungkvist
 * @date 2016
 */
template <int dim, typename Number>
class MGTransferMatrixFreeGpu : public MGTransferBase<GpuVector<Number> >
{
public:
  /**
   * Constructor without constraint matrices. Use this constructor only with
   * discontinuous finite elements or with no local refinement.
   */
  MGTransferMatrixFreeGpu ();

  /**
   * Constructor with constraints. Equivalent to the default constructor
   * followed by initialize_constraints().
   */
  MGTransferMatrixFreeGpu (const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Destructor.
   */
  virtual ~MGTransferMatrixFreeGpu ();

  /**
   * Initialize the constraints to be used in build().
   */
  void initialize_constraints (const MGConstrainedDoFs &mg_constrained_dofs);

  /**
   * Reset the object to the state it had right after the default constructor.
   */
  void clear ();

  /**
   * Actually build the information for the prolongation for each level.
   */
  void build (const DoFHandler<dim,dim> &mg_dof);

  /**
   * Prolongate a vector from level <tt>to_level-1</tt> to level
   * <tt>to_level</tt> using the embedding matrices of the underlying finite
   * element. The previous content of <tt>dst</tt> is overwritten.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the coarser level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * finer level.
   */
  virtual void prolongate (const unsigned int      to_level,
                           GpuVector<Number>       &dst,
                           const GpuVector<Number> &src) const;

  /**
   * Restrict a vector from level <tt>from_level</tt> to level
   * <tt>from_level-1</tt> using the transpose operation of the prolongate()
   * method. If the region covered by cells on level <tt>from_level</tt> is
   * smaller than that of level <tt>from_level-1</tt> (local refinement), then
   * some degrees of freedom in <tt>dst</tt> are active and will not be
   * altered. For the other degrees of freedom, the result of the restriction
   * is added.
   *
   * @param src is a vector with as many elements as there are degrees of
   * freedom on the finer level involved.
   *
   * @param dst has as many elements as there are degrees of freedom on the
   * coarser level.
   */
  virtual void restrict_and_add (const unsigned int from_level,
                                 GpuVector<Number>       &dst,
                                 const GpuVector<Number> &src) const;

  /**
   * Transfer from multi-level vector to normal vector.
   *
   * Copies data from active portions of an MGVector into the respective
   * positions of a <tt>Vector<number></tt>. In order to keep the result
   * consistent, constrained degrees of freedom are set to zero.
   */
  template <int spacedim>
  void
  copy_to_mg (const DoFHandler<dim,spacedim>    &mg_dof,
              MGLevelObject<GpuVector<Number> > &dst,
              const GpuVector<Number>           &src) const;

  /**
   * Transfer from multi-level vector to normal vector.
   *
   * Copies data from active portions of an MGVector into the respective
   * positions of a <tt>Vector<number></tt>. In order to keep the result
   * consistent, constrained degrees of freedom are set to zero.
   */
  template <int spacedim>
  void
  copy_from_mg (const DoFHandler<dim,spacedim>         &mg_dof,
                GpuVector<Number>                      &dst,
                const MGLevelObject<GpuVector<Number>> &src) const;

  /**
   * Add a multi-level vector to a normal vector.
   *
   * Works as the previous function, but probably not for continuous elements.
   */
  template <int spacedim>
  void
  copy_from_mg_add (const DoFHandler<dim,spacedim>         &mg_dof,
                    GpuVector<Number>                      &dst,
                    const MGLevelObject<GpuVector<Number>> &src) const;


  /**
   * Finite element does not provide prolongation matrices.
   */
  DeclException0(ExcNoProlongation);

  /**
   * Memory used by this object.
   */
  std::size_t memory_consumption () const;

private:

  /**
   * Stores the degree of the finite element contained in the DoFHandler
   * passed to build(). The selection of the computational kernel is based on
   * this number.
   */
  unsigned int fe_degree;

  /**
   * Stores whether the element is continuous and there is a joint degree of
   * freedom in the center of the 1D line.
   */
  bool element_is_continuous;

  /**
   * Stores the number of components in the finite element contained in the
   * DoFHandler passed to build().
   */
  unsigned int n_components;

  /**
   * Stores the number of degrees of freedom on all child cells. It is
   * <tt>2<sup>dim</sup>*fe.dofs_per_cell</tt> for DG elements and somewhat
   * less for continuous elements.
   */
  unsigned int n_child_cell_dofs;

  /**
   * Holds the indices for cells on a given level, extracted from DoFHandler
   * for fast access. All DoF indices on a given level are stored as a plain
   * array (since this class assumes constant DoFs per cell). To index into
   * this array, use the cell number times dofs_per_cell.
   *
   * This array first is arranged such that all locally owned level cells come
   * first (found in the variable n_owned_level_cells) and then other cells
   * necessary for the transfer to the next level.
   */

  // FIXME: change to GpuVector or similar class
  std::vector<GpuList<unsigned int> > level_dof_indices;

  /**
   * Stores the connectivity from parent to child cell numbers for each level.
   */
  // std::vector<std::vector<std::pair<unsigned int,unsigned int> > > parent_child_connect;
  std::vector<GpuList<unsigned int> > child_offset_in_parent;

  /**
   * Stores the number of cells owned on a given process (sets the bounds for
   * the worker loops) for each level.
   */
  std::vector<unsigned int> n_owned_level_cells;

  /**
   * Holds the one-dimensional embedding (prolongation) matrix from mother
   * element to the children.
   */
  // internal::MatrixFreeFunctions::ShapeInfo<Number> shape_info;
  GpuVector<Number> shape_values;

  /**
   * Holds mappings between global DoFs and level-local DoFs, stored in a structure-of-array format
   */
  std::vector<internal::IndexMapping> copy_indices;

  /**
   * Stores whether the copy operation from the global to the level vector is
   * actually a plain copy to the finest level. This means that the grid has
   * no adaptive refinement and the numbering on the finest multigrid level is
   * the same as in the global case.
   */
  bool perform_plain_copy;

  /**
   * For continuous elements, restriction is not additive and we need to
   * weight the result at the end of prolongation (and at the start of
   * restriction) by the valence of the degrees of freedom, i.e., on how many
   * elements they appear. We store the data in vectorized form to allow for
   * cheap access. Moreover, we utilize the fact that we only need to store
   * <tt>3<sup>dim</sup></tt> indices.
   *
   * Data is organized in terms of each level (outer vector) and the cells on
   * each level (inner vector).
   */
  std::vector<GpuVector<Number> > weights_on_refined;

  /**
   * Stores the local indices of Dirichlet boundary conditions on cells for
   * all levels (outer index), the cells within the levels (second index), and
   * the indices on the cell (inner index).
   */
  // std::vector<std::vector<std::vector<unsigned short> > > dirichlet_indices;
  std::vector<GpuList<unsigned int> > dirichlet_indices;

  /**
   * Internal function for setting up data structures for solution transfer
   */
  void setup_copy_indices(const DoFHandler<dim,dim>  &mg_dof);

  /**
   * Performs templated prolongation operation
   */
  template <int degree>
  void do_prolongate_add(const unsigned int       to_level,
                         GpuVector<Number>       &dst,
                         const GpuVector<Number> &src) const;

  /**
   * Performs templated restriction operation
   */
  template <int degree>
  void do_restrict_add(const unsigned int       from_level,
                       GpuVector<Number>       &dst,
                       const GpuVector<Number> &src) const;

  template <template <int,int,typename> class loop_body, int degree>
  void coarse_cell_loop  (const unsigned int      coarse_level,
                          GpuVector<Number>       &dst,
                          const GpuVector<Number> &src) const;

  void set_constrained_dofs(GpuVector<Number>& vec, unsigned int level, Number val) const;

  SmartPointer< const MGConstrainedDoFs, MGTransferMatrixFreeGpu<dim,Number> > 	mg_constrained_dofs;

};


/*@}*/


DEAL_II_NAMESPACE_CLOSE

#endif
