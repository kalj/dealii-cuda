#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>

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

  GridGenerator::hyper_cube (triangulation, 0., 1.);

  triangulation.refine_global(2);

  dof_handler.distribute_dofs (fe);
  dof_handler.distribute_mg_dofs (fe);


  MGTransferMatrixFreeGpu<dimension,Number> mg;


  mg.clear();

  mg.build(dof_handler);

  mg.clear();

  return 0;
}
