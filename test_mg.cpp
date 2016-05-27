#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/lac/vector.h>

#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/gpu_vec.h"

using namespace dealii;

int main(int argc, char **argv)
{

  typedef double Number;
  const unsigned int dimension = 2;
  const unsigned int fe_degree = 1;

  Triangulation<dimension>               triangulation;
  FE_Q<dimension>                        fe(fe_degree);
  DoFHandler<dimension>                  dof_handler(triangulation);
  // Quadrature<dimension> q(fe.get_unit_support_points());
  // FEValues<dimension> fe_values(fe,q,update_q_points);

  GridGenerator::subdivided_hyper_cube (triangulation, 4);

  triangulation.refine_global(1);


  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);


  ConstraintMatrix hanging_node_constraints;
  MGConstrainedDoFs mg_constrained_dofs;
  ZeroFunction<dimension> zero_function;
  typename FunctionMap<dimension>::type dirichlet_boundary;
  dirichlet_boundary[0] = &zero_function;
  mg_constrained_dofs.initialize(dof_handler, dirichlet_boundary);


  // build reference
  MGTransferPrebuilt<Vector<double> >
    transfer_ref(hanging_node_constraints, mg_constrained_dofs);
  transfer_ref.build_matrices(dof_handler);

  // build matrix-free transfer
  MGTransferMatrixFreeGpu<dimension,Number> transfer(mg_constrained_dofs);
  transfer.build(dof_handler);


  // check prolongation for all levels using random vector
  for (unsigned int level=1; level<dof_handler.get_triangulation().n_global_levels(); ++level)
  {
    Vector<Number> v1, v2;
    GpuVector<Number> v1_dev, v2_dev;
    Vector<double> v1_cpy, v2_cpy, v3;
    v1.reinit(dof_handler.n_dofs(level-1));
    v2.reinit(dof_handler.n_dofs(level));
    v3.reinit(dof_handler.n_dofs(level));
    for (unsigned int i=0; i<v1.size(); ++i)
      v1[i] = (double)rand()/RAND_MAX;

    v1_dev = v1;
    v2_dev = v2;
    transfer.prolongate(level, v2_dev, v1_dev);
    v2_dev.copyToHost(v2);

    v1_cpy = v1;
    transfer_ref.prolongate(level, v3, v1_cpy);

    v2_cpy = v2;
    v3 -= v2_cpy;
    std::cout << "Diff prolongate   l" << level << ": " << v3.l2_norm() << std::endl;
  }

  // check restriction for all levels using random vector
  for (unsigned int level=1; level<dof_handler.get_triangulation().n_global_levels(); ++level)
  {
    Vector<Number> v1, v2;
    GpuVector<Number> v1_dev, v2_dev;
    Vector<double> v1_cpy, v2_cpy, v3;
    v1.reinit(dof_handler.n_dofs(level));
    v2.reinit(dof_handler.n_dofs(level-1));
    v3.reinit(dof_handler.n_dofs(level-1));
    for (unsigned int i=0; i<v1.size(); ++i)
      v1[i] = (double)rand()/RAND_MAX;

    v1_dev = v1;
    v2_dev = v2;
    transfer.restrict_and_add(level, v2_dev, v1_dev);
    v2_dev.copyToHost(v2);

    v1_cpy = v1;
    transfer_ref.restrict_and_add(level, v3, v1_cpy);

    v2_cpy = v2;
    v3 -= v2_cpy;
    std::cout << "Diff restrict     l" << level << ": " << v3.l2_norm() << std::endl;

    v2 = 1.;
    v3 = 1.;

    v1_dev = v1;
    v2_dev = v2;
    transfer.restrict_and_add(level, v2_dev, v1_dev);

    transfer_ref.restrict_and_add(level, v3, v1_cpy);
    v2_dev.copyToHost(v2);

    v2_cpy = v2;
    v3 -= v2_cpy;
    std::cout << "Diff restrict add l" << level << ": " << v3.l2_norm() << std::endl;
  }


  return 0;
}
