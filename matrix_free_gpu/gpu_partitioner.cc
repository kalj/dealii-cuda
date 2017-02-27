/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

#include "gpu_partitioner.h"
#include <deal.II/fe/fe.h>

using namespace dealii;

GpuPartitioner::GpuPartitioner()
{}

template <int dim>
GpuPartitioner::GpuPartitioner(const DoFHandler<dim> &dof_handler,
                                    unsigned int num_partitions)
{
  reinit(dof_handler,num_partitions);
}

template <int dim>
void GpuPartitioner::reinit(const DoFHandler<dim> &dof_handler,
                            unsigned int num_partitions)
{
  n_parts = num_partitions;

  local_dof_offsets.resize(n_parts);
  local_cell_offsets.resize(n_parts);
  n_local_dofs.resize(n_parts);
  n_local_cells.resize(n_parts);
  ghost_dof_indices.resize(n_parts);

  n_ghost_dofs_matrix.resize(n_parts);

  for(int i = 0; i < n_parts; ++i)
    n_ghost_dofs_matrix[i].resize(n_parts);

  n_import_inds.resize(n_parts);
  import_inds.resize(n_parts);

  n_cells_tot = dof_handler.get_triangulation().n_active_cells();
  n_dofs_tot = dof_handler.n_dofs();


  //---------------------------------------------------------------------------
  // set up cell partitions
  //---------------------------------------------------------------------------
  unsigned int chunk_size = 1 + (n_cells_tot-1)/n_parts;

  for(int i = 0; i < n_parts; ++i) {
    local_cell_offsets[i] = i*chunk_size;
    n_local_cells[i] = std::min(chunk_size,
                                n_cells_tot-local_cell_offsets[i]);

  }

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

  n_local_dofs[n_parts-1] = n_dofs_tot - local_dof_offsets[n_parts-1];


  //---------------------------------------------------------------------------
  // set up ghosted dofs
  //---------------------------------------------------------------------------
  std::set<unsigned int> gathered;
  for(int i = 0; i < n_parts; ++i) {
    gathered.clear();

    for(typename DoFHandler<dim>::active_cell_iterator cell=this->begin_active(dof_handler,i);
        cell!=this->end(dof_handler,i);
        ++cell) {

      cell->get_dof_indices(dof_indices);
      for(auto idx : dof_indices) {
        if(idx < local_dof_offsets[i]) {
          gathered.insert(idx);
        }
      }
    }

    // all are gathered now
    ghost_dof_indices[i] = std::vector<unsigned int>(gathered.begin(),gathered.end());

    // find size of regions belonging to each partition:
    // initialize to zero
    for(int j=0; j<n_parts; ++j)
      n_ghost_dofs_matrix[i][j]=0;

    // count ghost dofs from other
    for(auto g : ghost_dof_indices[i]) {
      unsigned int owner = dof_owner(g);
      n_ghost_dofs_matrix[i][owner]++;
    }

  }

  // set up lists of which dofs in a partition are ghosted in other partitions
  // these are indices in local index space. note that there may be duplicates
  // as several other partitions may have the same dof as ghost
  for(int owner = 0; owner < n_parts; ++owner) {

    // set up size
    n_import_inds[owner] = 0;
    for(int other = 0; other < n_parts; ++other)
      n_import_inds[owner] += n_ghost_dofs_matrix[other][owner];

    import_inds[owner].resize(n_import_inds[owner]);

    for(int other = 0; other < n_parts; ++other) {

      for(int i=0; i<n_ghost_dofs_matrix[other][owner]; ++i)
        import_inds[owner][this->import_data_offset(owner,other)+i]
          = ghost_dof_indices[other][this->ghost_dofs_offset(other,owner)+i]
          - local_dof_offsets[owner];
    }
  }

}

unsigned int GpuPartitioner::n_partitions() const
{
  return n_parts;
}

unsigned int GpuPartitioner::n_dofs(unsigned int part) const
{
  return n_local_dofs[part];
}


unsigned int GpuPartitioner::n_global_dofs() const
{
  return n_dofs_tot;
}

void GpuPartitioner::extract_locally_relevant_dofs(const unsigned int part,
                                                   IndexSet &index_set) const
{
  index_set.clear();
  index_set.set_size (n_dofs_tot);

  index_set.add_range (local_dof_offsets[part],
                       local_dof_offsets[part]+n_local_dofs[part]);
}

unsigned int GpuPartitioner::n_cells(unsigned int part) const
{
  return n_local_cells[part];
}

unsigned int GpuPartitioner::n_global_cells() const
{
  return n_cells_tot;
}

unsigned int GpuPartitioner::local_dof_offset(unsigned int part) const
{
  return local_dof_offsets[part];
}

bool GpuPartitioner::is_compatible(const GpuPartitioner &other) const
{
  return this == &other;
}

unsigned int GpuPartitioner::dof_owner(unsigned int global_index) const
{
  unsigned int owner = 0;
  for(int i=i; i< n_parts; ++i) {
    if(global_index >= local_dof_offsets[i])
      owner = i;
  }

  return owner;
}

unsigned int GpuPartitioner::cell_owner(unsigned int cell_index) const
{
  unsigned int owner = 0;
  for(int i=i; i< n_parts; ++i) {
    if(cell_index >= local_cell_offsets[i])
      owner = i;
  }

  return owner;
}


unsigned int GpuPartitioner::local_index(unsigned int part,
                                         unsigned int global_index) const
{
  const unsigned int owner = dof_owner(global_index);
  unsigned int l = numbers::invalid_unsigned_int;
  if(owner == part)
    l = global_index - local_dof_offsets[part];
  else {

    const unsigned int offset = ghost_dofs_offset(part,owner);
    for(int i = 0; i < n_ghost_dofs_matrix[part][owner]; ++i) {
      if(ghost_dof_indices[part][offset+i] == global_index) {
        l = n_local_dofs[part]+offset+i;
        break;
      }
    }
  }
  // we went through all ghost dofs without matches
  Assert(l != numbers::invalid_unsigned_int,
         ExcInternalError("No such dof index found in current partition and its ghosts"));

  return l;
}

template <int dim>
typename DoFHandler<dim>::active_cell_iterator GpuPartitioner::begin_active(const DoFHandler<dim> &dof_handler,
                                                                            unsigned int part) const
{
  return typename DoFHandler<dim>::active_cell_iterator (&dof_handler.get_triangulation(),
                                                         dof_handler.get_triangulation().n_levels()-1,
                                                         local_cell_offsets[part],
                                                         &dof_handler);
}

template <int dim>
typename DoFHandler<dim>::cell_iterator GpuPartitioner::end(const DoFHandler<dim> &dof_handler,
                                                            unsigned int part) const
{
  if(part<n_parts-1)
    return typename DoFHandler<dim>::cell_iterator (&dof_handler.get_triangulation(),
                                                    dof_handler.get_triangulation().n_levels()-1,
                                                    local_cell_offsets[part]+n_local_cells[part],
                                                    &dof_handler);
  else
    return dof_handler.end();
}



template <int dim>
typename DoFHandler<dim>::level_cell_iterator GpuPartitioner::begin_mg(const DoFHandler<dim> &dof_handler,
                                                                       unsigned int partition,
                                                                       unsigned int level) const
{
  // FIXME: Implement this!
  AssertThrow(false,dealii::ExcNotImplemented());
  return dof_handler.begin_mg(level);
}

template <int dim>
typename DoFHandler<dim>::level_cell_iterator GpuPartitioner::end_mg(const DoFHandler<dim> &dof_handler,
                                                                     unsigned int partition,
                                                                     unsigned int level) const
{
  // FIXME: Implement this!
  AssertThrow(false,dealii::ExcNotImplemented());
  return dof_handler.end_mg(level);
}


const std::vector<unsigned int>& GpuPartitioner::import_indices(unsigned int owner) const
{
  return import_inds[owner];
}

const std::vector<std::vector<unsigned int> >& GpuPartitioner::import_indices() const
{
  return import_inds;
}

unsigned int GpuPartitioner::n_import_indices(unsigned int owner) const
{
  return n_import_inds[owner];
}

const std::vector<unsigned int>& GpuPartitioner::n_import_indices() const
{
  return n_import_inds;
}

unsigned int GpuPartitioner::import_data_offset(unsigned int owner,
                                                unsigned int with_ghost) const
{
  // Assert(owner != with_ghost); // ghosts with oneself don't make sense
  unsigned int offset=0;
  for(int i = 0; i < with_ghost; ++i)
    offset += n_ghost_dofs_matrix[i][owner];

  return offset;
}

unsigned int GpuPartitioner::ghost_dofs_offset(unsigned int with_ghost,
                                               unsigned int owner) const
{
  // Assert(owner != with_ghost); // ghosts with oneself don't make sense
  unsigned int offset=0;
  for(int i = 0; i < owner; ++i)
    offset += n_ghost_dofs_matrix[with_ghost][i];

  return offset;
}

unsigned int GpuPartitioner::n_ghost_dofs(unsigned int with_ghost,
                                          unsigned int owner) const
{
  // Assert(owner != with_ghost); // ghosts with oneself don't make sense

  return n_ghost_dofs_matrix[with_ghost][owner];
}

unsigned int GpuPartitioner::n_ghost_dofs_tot(unsigned int with_ghost) const
{
  unsigned int n=0;
  for(int i = 0; i < n_parts; ++i)
    n += n_ghost_dofs_matrix[with_ghost][i];
  return n;
}

const std::vector<unsigned int>& GpuPartitioner::ghost_indices(unsigned int with_ghost) const
{
  return ghost_dof_indices[with_ghost];
}


//=============================================================================
// explicit instantiations
//=============================================================================

// dim==2
template GpuPartitioner::GpuPartitioner(const DoFHandler<2> &dof_handler,
                                        unsigned int num_partitions);
template
void GpuPartitioner::reinit(const DoFHandler<2> &dof_handler,
                            unsigned int num_partitions);
template
typename DoFHandler<2>::active_cell_iterator GpuPartitioner::begin_active(const DoFHandler<2> &dof_handler,
                                                                          unsigned int partition) const;
template
typename DoFHandler<2>::cell_iterator GpuPartitioner::end(const DoFHandler<2> &dof_handler,
                                                          unsigned int partition) const;

template
typename DoFHandler<2>::level_cell_iterator GpuPartitioner::begin_mg(const DoFHandler<2> &dof_handler,
                                                                     unsigned int partition,
                                                                     unsigned int level) const;
template
typename DoFHandler<2>::level_cell_iterator GpuPartitioner::end_mg(const DoFHandler<2> &dof_handler,
                                                                     unsigned int partition,
                                                                     unsigned int level) const;

// dim==3
template GpuPartitioner::GpuPartitioner(const DoFHandler<3> &dof_handler,
                                        unsigned int num_partitions);
template
void GpuPartitioner::reinit(const DoFHandler<3> &dof_handler,
                            unsigned int num_partitions);

template
typename DoFHandler<3>::active_cell_iterator GpuPartitioner::begin_active(const DoFHandler<3> &dof_handler,
                                                                          unsigned int partition) const;
template
typename DoFHandler<3>::cell_iterator GpuPartitioner::end(const DoFHandler<3> &dof_handler,
                                                          unsigned int partition) const;

template
typename DoFHandler<3>::level_cell_iterator GpuPartitioner::begin_mg(const DoFHandler<3> &dof_handler,
                                                                     unsigned int partition,
                                                                     unsigned int level) const;
template
typename DoFHandler<3>::level_cell_iterator GpuPartitioner::end_mg(const DoFHandler<3> &dof_handler,
                                                                     unsigned int partition,
                                                                     unsigned int level) const;
