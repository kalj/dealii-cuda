#ifndef dealii__constraint_handler_gpu_h
#define dealii__constraint_handler_gpu_h

#include <deal.II/base/config.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include "multi_gpu_vec.h"
#include "multi_gpu_list.h"

DEAL_II_NAMESPACE_OPEN


template <typename Number>
class ConstraintHandlerGpu
{
public:

  // ConstraintHandlerGpu();

  void reinit(const std::vector<ConstraintMatrix> &constraints,
              const std::shared_ptr<const GpuPartitioner> &partitioner);

  void reinit(const std::vector<MGConstrainedDoFs> &mg_constrained_dofs,
         const std::shared_ptr<const GpuPartitioner> &partitioner,
              const unsigned int        level);

  void set_constrained_values(MultiGpuVector<Number>        &v,
                              Number val) const;

  void save_constrained_values(MultiGpuVector<Number>        &v);

  void save_constrained_values(const MultiGpuVector <Number> &v1,
                               MultiGpuVector<Number>        &v2);

  void load_constrained_values(MultiGpuVector <Number>          &v) const;

  void load_and_add_constrained_values(MultiGpuVector<Number>          &v1,
                                       MultiGpuVector<Number>           &v2) const;

  void copy_edge_values(MultiGpuVector<Number> &dst, const MultiGpuVector<Number> &src) const;

  std::size_t memory_consumption () const;

private:
  void reinit_kernel_parameters();

  std::shared_ptr<const GpuPartitioner> partitioner;

  unsigned int                          n_partitions;

  std::vector<unsigned int>             n_constrained_dofs;

  // index lists
  MultiGpuList<unsigned int>            constrained_indices;
  MultiGpuList<unsigned int>            edge_indices;

  // temporary buffers
  MultiGpuList<Number>                  constrained_values_src;
  MultiGpuList<Number>                  constrained_values_dst;

  std::vector<dim3>                     grid_dim;
  dim3                                  block_dim;
};

DEAL_II_NAMESPACE_CLOSE

#endif
