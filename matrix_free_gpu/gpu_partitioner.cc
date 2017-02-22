/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include "gpu_partitioner.h"
#include <deal.II/fe/fe.h>

using namespace dealii;

template <int dim>
GpuPartitioner<dim>::GpuPartitioner()
{}

template <int dim>
GpuPartitioner<dim>::GpuPartitioner(const DoFHandler<dim> &dof_handler,
                                    unsigned int num_partitions)
{
  reinit(dof_handler,num_partitions);
}

template <int dim>
void GpuPartitioner<dim>::reinit(const DoFHandler<dim> &dof_handler,
                                 unsigned int num_partitions)
{
  n_parts = num_partitions;

  local_dof_offsets.resize(n_parts);
  local_cell_offsets.resize(n_parts);
  n_local_dofs.resize(n_parts);
  n_local_cells.resize(n_parts);
  ghost_dof_indices.resize(n_parts);
  n_local_dofs_with_ghosted.resize(n_parts);
  active_begin_iterators.resize(n_parts);
  end_iterators.resize(n_parts);

  n_ghost_dofs.resize(n_parts);

  n_global_cells = dof_handler.get_triangulation().n_active_cells();
  n_global_dofs = dof_handler.n_dofs();


  //---------------------------------------------------------------------------
  // set up cell partitions
  //---------------------------------------------------------------------------
  unsigned int chunk_size = 1 + (n_global_cells-1)/n_parts;

  for(int i = 0; i < n_parts; ++i) {
    local_cell_offsets[i] = i*chunk_size;
    n_local_cells[i] = std::min(chunk_size,
                                n_global_cells-local_cell_offsets[i]);

    active_begin_iterators[i] = typename DoFHandler<dim>::active_cell_iterator (&dof_handler.get_triangulation(),
                                                                                dof_handler.get_triangulation().n_levels()-1,
                                                                                local_cell_offsets[i],
                                                                                &dof_handler);
    if(i<n_parts-1) {
      end_iterators[i] = typename DoFHandler<dim>::cell_iterator (&dof_handler.get_triangulation(),
                                                                  dof_handler.get_triangulation().n_levels()-1,
                                                                  local_cell_offsets[i]+n_local_cells[i],
                                                                  &dof_handler);
    }
  }

  end_iterators[n_parts-1] = dof_handler.end();

  //---------------------------------------------------------------------------
  // set up resulting dof partitions
  //---------------------------------------------------------------------------
  const FiniteElement<dim> &fe = dof_handler.get_fe();
  const unsigned int dofs_per_cell = fe.dofs_per_cell;

  local_dof_offsets[0] = 0;

  std::vector< unsigned int > dof_indices(dofs_per_cell);
  for(int i = 1; i < n_parts; ++i) {
    const typename DoFHandler<dim>::active_cell_iterator last(&dof_handler.get_triangulation(),
                                                              dof_handler.get_triangulation().n_levels()-1,
                                                              local_cell_offsets[i-1]+n_local_cells[i-1]-1,
                                                              &dof_handler);

    last->get_dof_indices(dof_indices);

    unsigned int max_dof_idx = 0;
    for(auto idx : dof_indices)
      max_dof_idx = std::max(max_dof_idx,idx);

    local_dof_offsets[i] = max_dof_idx+1;

    n_local_dofs[i-1] = local_dof_offsets[i] - local_dof_offsets[i-1];
  }

  n_local_dofs[n_parts-1] = n_global_dofs - local_dof_offsets[n_parts-1];


  //---------------------------------------------------------------------------
  // set up ghosted dofs
  //---------------------------------------------------------------------------
  std::set<unsigned int> gathered;
  for(int i = 0; i < n_parts; ++i) {
    gathered.clear();

    for(typename DoFHandler<dim>::active_cell_iterator cell=active_begin_iterators[i];
        cell!=end_iterators[i]; ++cell) {

      cell->get_dof_indices(dof_indices);
      for(auto idx : dof_indices) {
        if(idx < local_dof_offsets[i]) {
          gathered.insert(idx);
        }
      }
    }

    // all are gathered now
    ghost_dof_indices[i] = std::vector<unsigned int>(gathered.begin(),gathered.end());
    // n_local_dofs_with_ghosted[i] = n_local_dofs[i] + ghost_dof_indices[i].size();
    n_ghost_dofs[i] = ghost_dof_indices[i].size();
  }

}

template <int dim>
unsigned int GpuPartitioner<dim>::n_partitions() const
{
  return n_parts;
}

template <int dim>
const std::vector<unsigned int>& GpuPartitioner<dim>::local_sizes() const
{
  return n_local_dofs;
}

// template <int dim>
// const std::vector<unsigned int>& GpuPartitioner<dim>::ghosted_sizes() const
// {
//   return n_local_dofs_with_ghosted;
// }

template <int dim>
unsigned int GpuPartitioner<dim>::global_size() const
{
  return n_global_dofs;
}

template <int dim>
unsigned int GpuPartitioner<dim>::local_dof_offset(unsigned int part) const
{
  return local_dof_offsets[part];
}

template <int dim>
bool GpuPartitioner<dim>::is_compatible(const GpuPartitioner &other) const
{

}

template <int dim>
unsigned int GpuPartitioner<dim>::dof_owner(unsigned int global_index) const
{
  unsigned int owner = 0;
  while (owner < n_parts && local_dof_offsets[owner+1] <= global_index) {
    owner++;
  }

  return owner;
}

template <int dim>
unsigned int GpuPartitioner<dim>::cell_owner(unsigned int cell_index) const
{
  unsigned int owner = 0;
  while (owner < n_parts && local_cell_offsets[owner+1] <= cell_index) {
    owner++;
  }

  return owner;
}

template <int dim>
unsigned int GpuPartitioner<dim>::local_index(unsigned int global_index) const
{
  return global_index - local_dof_offsets[dof_owner(global_index)];
}

template <int dim>
typename DoFHandler<dim>::active_cell_iterator GpuPartitioner<dim>::begin_active(unsigned part)
{
  return active_begin_iterators[part];
}

template <int dim>
typename DoFHandler<dim>::cell_iterator GpuPartitioner<dim>::end(unsigned part)
{
  return end_iterators[part];
}

template class GpuPartitioner<2>;
template class GpuPartitioner<3>;
