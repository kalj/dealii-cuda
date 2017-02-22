#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <sstream>

// #include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_partitioner.h"

using namespace dealii;

int main(int argc, char *argv[])
{
  const int dim = 2;
  const int fe_degree = 2;

  FE_Q<dim>                        fe(fe_degree);
  Triangulation<dim>               triangulation;
  DoFHandler<dim>                  dof_handler(triangulation);

  GridGenerator::hyper_cube (triangulation, -1., 1.);
  triangulation.refine_global(5);

  dof_handler.distribute_dofs (fe);

  const int n_partitions = 8;
  GpuPartitioner<dim> partitioner(dof_handler,n_partitions);

  std::cout << "Partition in " << n_partitions << " parts" << std::endl;
  for(int i = 0; i < partitioner.n_partitions(); ++i) {
    std::cout << " Partition " << i << ":" << std::endl;
    std::cout << "   # owned cells: " << partitioner.n_local_cells[i] << std::endl;
    std::cout << "   # owned dofs:  " << partitioner.n_local_dofs[i] << std::endl;
    std::cout << "   # ghost dofs:  " << partitioner.n_ghost_dofs[i] << std::endl;
    std::cout << "   local_cell_offsets: " << partitioner.local_cell_offsets[i] << std::endl;
    std::cout << "   local_dof_offsets: " << partitioner.local_dof_offsets[i] << std::endl;
  }


  Vector<double> label(triangulation.n_active_cells());
  for(int c = 0; c < triangulation.n_active_cells(); ++c) {
    label[c] = partitioner.cell_owner(c);
  }

  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (label, "label");
  data_out.build_patches ();

  std::ostringstream filename;
  filename << "partition"
           << ".vtu";

  std::ofstream output (filename.str().c_str());
  data_out.write_vtu (output);




  return 0;
}
