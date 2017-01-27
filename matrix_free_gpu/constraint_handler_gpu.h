#ifndef dealii__constraint_handler_gpu_h
#define dealii__constraint_handler_gpu_h

#include <deal.II/base/config.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include "gpu_vec.h"
#include "gpu_list.h"

DEAL_II_NAMESPACE_OPEN


template <typename Number>
class ConstraintHandlerGpu
{
public:

  // ConstraintHandlerGpu();

  void reinit(const ConstraintMatrix &constraints,
              const unsigned int n_dofs);

  void reinit(const MGConstrainedDoFs  &mg_constrained_dofs,
              const unsigned int        level);


  void set_constrained_values(GpuVector<Number>        &v,
                              Number val) const;

  void save_constrained_values(GpuVector<Number>        &v);

  void save_constrained_values(const GpuVector <Number> &v1,
                               GpuVector<Number>        &v2);

  void load_constrained_values(GpuVector <Number>          &v) const;

  void load_and_add_constrained_values(GpuVector<Number>          &v1,
                                       GpuVector<Number>           &v2) const;

  void copy_edge_values(GpuVector<Number> &dst, const GpuVector<Number> &src) const;

  std::size_t memory_consumption () const;

private:
  void reinit_kernel_parameters();

  unsigned int                 n_constrained_dofs;

  // index lists
  GpuList<unsigned int>        constrained_indices;
  GpuList<unsigned int>        edge_indices;

  // temporary buffers
  GpuVector<Number>                   constrained_values_src;
  GpuVector<Number>                   constrained_values_dst;

  dim3 grid_dim;
  dim3 block_dim;
};

DEAL_II_NAMESPACE_CLOSE

#endif
