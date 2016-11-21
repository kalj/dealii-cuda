#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.h>

#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/gpu_vec.h"

using namespace dealii;

template <typename HostVectorT, typename Number>
double compute_l2_norm(const HostVectorT& v_host, const GpuVector<Number> &v_dev)
{
  Vector<Number> v_dev_host(v_dev.size());
  v_dev.copyToHost(v_dev_host);

  double diff = 0;
  for(int i = 0; i < v_host.size(); ++i) {
    double d = v_dev_host[i]-v_host[i];
    diff += d*d;
  }
  return sqrt(diff);
}

template <int dim>
void check(int fe_degree)
{
  std::cout << "Running tests for dim=" << dim << ", fe_degree="<<fe_degree << std::endl;
  typedef double Number;
  const int nref = 4;

  Triangulation<dim>  triangulation(Triangulation<dim>::limit_level_difference_at_vertices);
  FE_Q<dim>           fe(fe_degree);
  DoFHandler<dim>     dof_handler(triangulation);

  GridGenerator::subdivided_hyper_cube (triangulation, 4);

  triangulation.refine_global(nref);


  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);

  MGConstrainedDoFs mg_constrained_dofs;

  mg_constrained_dofs.initialize(dof_handler);
  std::set<types::boundary_id> bdry;
  bdry.insert(0);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,bdry);


  // build reference
  MGTransferMatrixFree<dim,Number> transfer_ref(mg_constrained_dofs);
  transfer_ref.build(dof_handler);

  // build matrix-free transfer
  MGTransferMatrixFreeGpu<dim,Number> transfer(mg_constrained_dofs);
  transfer.build(dof_handler);


  // check prolongation for all levels using random vector
  for (unsigned int level=1; level<dof_handler.get_triangulation().n_global_levels(); ++level)
  {
    int n_dofs_src = dof_handler.n_dofs(level-1);
    int n_dofs_dst = dof_handler.n_dofs(level);

    LinearAlgebra::distributed::Vector<Number> v_src_host(n_dofs_src);
    LinearAlgebra::distributed::Vector<Number> v_dst_host(n_dofs_dst);
    GpuVector<Number> v_src_dev(n_dofs_src);
    GpuVector<Number> v_dst_dev(n_dofs_dst);

    {
      Vector<Number> v_src_dev_init(n_dofs_src);
      for (unsigned int i=0; i<n_dofs_src; ++i) {
        v_src_host[i] = (double)rand()/RAND_MAX;
        v_src_dev_init[i] = (double)rand()/RAND_MAX;
      }
      v_src_dev = v_src_dev_init;
    }

    transfer_ref.prolongate(level, v_dst_host, v_src_host);

    transfer.prolongate(level, v_dst_dev, v_src_dev);

    std::cout << "Diff prolongate   l" << level << ": " << compute_l2_norm(v_dst_host,v_dst_dev) << std::endl;
  }

  // check restriction for all levels using random vector
  for (unsigned int level=1; level<dof_handler.get_triangulation().n_global_levels(); ++level)
  {
    int n_dofs_src = dof_handler.n_dofs(level);
    int n_dofs_dst = dof_handler.n_dofs(level-1);

    LinearAlgebra::distributed::Vector<Number> v_src_host(n_dofs_src);
    LinearAlgebra::distributed::Vector<Number> v_dst_host(n_dofs_dst);
    GpuVector<Number> v_src_dev(n_dofs_src);
    GpuVector<Number> v_dst_dev(n_dofs_dst);

    {
      Vector<Number> v_src_dev_init(n_dofs_src);
      for (unsigned int i=0; i<n_dofs_src; ++i) {
        v_src_host[i] = (double)rand()/RAND_MAX;
        v_src_dev_init[i] = (double)rand()/RAND_MAX;
      }
      v_src_dev = v_src_dev_init;
    }

    v_dst_host = 0.;
    v_dst_dev  = 0.;

    transfer_ref.restrict_and_add(level, v_dst_host, v_src_host);

    transfer.restrict_and_add(level, v_dst_dev, v_src_dev);

    std::cout << "Diff restrict     l" << level << ": " << compute_l2_norm(v_dst_host,v_dst_dev) << std::endl;

    v_dst_host = 1.;
    v_dst_dev  = 1.;

    transfer_ref.restrict_and_add(level, v_dst_host, v_src_host);

    transfer.restrict_and_add(level, v_dst_dev, v_src_dev);

    std::cout << "Diff restrict add l" << level << ": " << compute_l2_norm(v_dst_host,v_dst_dev) << std::endl;

  }

}

int main(int argc, char **argv)
{
  check<2>(1);
  check<2>(2);
  check<2>(3);
  check<2>(4);

  check<3>(1);
  check<3>(2);
  check<3>(3);
  check<3>(4);

  return 0;
}
