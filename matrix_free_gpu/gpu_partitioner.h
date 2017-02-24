/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#ifndef _GPU_PARTITIONER_H
#define _GPU_PARTITIONER_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>

namespace dealii
{
  class GpuPartitioner
  {
  public:
    GpuPartitioner();

    template <int dim>
    GpuPartitioner(const DoFHandler<dim> &dof_handler, unsigned int n_partitions);

    template <int dim>
    void reinit(const DoFHandler<dim> &dof_handler, unsigned int n_partitions);

    unsigned int n_partitions() const;

    unsigned int n_global_dofs() const;

    unsigned int n_dofs(unsigned int part) const;

    unsigned int n_global_cells() const;

    unsigned int n_cells(unsigned int part) const;

    unsigned int local_dof_offset(unsigned int part) const;

    bool is_compatible(const GpuPartitioner &other) const;

    unsigned int cell_owner(unsigned int cell_index) const;

    unsigned int dof_owner(unsigned int global_index) const;

    unsigned int local_index(unsigned int global_index) const;

    const std::vector<unsigned int>& import_indices(unsigned int owner) const;

    const std::vector<unsigned int>& ghost_indices(unsigned int with_ghost) const;

    unsigned int n_import_indices(unsigned int owner) const;

    unsigned int import_data_offset(unsigned int owner,
                                    unsigned int with_ghost) const;

    unsigned int ghost_dofs_offset(unsigned int with_ghost,
                                   unsigned int owner) const;

    unsigned int n_ghost_dofs(unsigned int with_ghost,
                              unsigned int owner) const;

    unsigned int n_ghost_dofs_tot(unsigned int with_ghost) const;

    template <int dim>
    typename DoFHandler<dim>::active_cell_iterator begin_active(const DoFHandler<dim> &dof_handler,
                                                                unsigned int partition) const;

    template <int dim>
    typename DoFHandler<dim>::cell_iterator end(const DoFHandler<dim> &dof_handler,
                                                unsigned int partition) const;


  private:

    // partitions
    unsigned int                             n_parts;

    // related to cells
    unsigned int                             n_cells_tot;
    std::vector<unsigned int>                local_cell_offsets;
    std::vector<unsigned int>                n_local_cells;

    // related to dofs
    unsigned int                             n_dofs_tot;
    std::vector<unsigned int>                local_dof_offsets;
    std::vector<unsigned int>                n_local_dofs;

    // related do ghost dofs
    std::vector<std::vector<unsigned int> >  ghost_dof_indices;
    std::vector<std::vector<unsigned int> >  n_ghost_dofs_matrix;

    std::vector<std::vector<unsigned int> >  import_inds;
    std::vector<unsigned int >               n_import_inds;

  };
}

#endif /* _GPU_PARTITIONER_H */
