/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#ifndef _GPU_PARTITIONER_H
#define _GPU_PARTITIONER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

namespace dealii
{
  template <int dim>
  class GpuPartitioner
  {
  public:
    GpuPartitioner();

    GpuPartitioner(const DoFHandler<dim> &dof_handler, unsigned int n_partitions);

    void reinit(const DoFHandler<dim> &dof_handler, unsigned int n_partitions);

    unsigned int n_partitions() const;

    const std::vector<unsigned int>& local_sizes() const;

    const std::vector<unsigned int>& ghosted_sizes() const;

    unsigned int global_size() const;

    unsigned int local_dof_offset(unsigned int part) const;

    bool is_compatible(const GpuPartitioner &other) const;

    unsigned int cell_owner(unsigned int cell_index) const;

    unsigned int dof_owner(unsigned int global_index) const;

    unsigned int local_index(unsigned int global_index) const;

    typename DoFHandler<dim>::active_cell_iterator begin_active(unsigned partition);

    typename DoFHandler<dim>::cell_iterator end(unsigned partition);

  // private:
    unsigned int n_parts;

    unsigned int n_global_dofs;

    unsigned int n_global_cells;

    std::vector<unsigned int> local_dof_offsets;

    std::vector<unsigned int> local_cell_offsets;

    std::vector<unsigned int> n_local_dofs;

    std::vector<unsigned int> n_local_cells;

    std::vector<std::vector<unsigned int> > ghost_dof_indices;

    std::vector<unsigned int> n_local_dofs_with_ghosted;

    std::vector<unsigned int> n_ghost_dofs;

    std::vector<typename DoFHandler<dim>::active_cell_iterator> active_begin_iterators;

    std::vector<typename DoFHandler<dim>::cell_iterator> end_iterators;

  };

}

#endif /* _GPU_PARTITIONER_H */
