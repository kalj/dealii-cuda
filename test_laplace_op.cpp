/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <sstream>

#include "gpu_vec.h"
#include "laplace_operator.h"
#include "laplace_operator_gpu.h"
#include "timing.h"

using namespace dealii;


const unsigned int degree_finite_element = 2;
const unsigned int dimension = 2;
const unsigned int Nreps = 500;

void assemble_matrix (SparseMatrix<double> &system_matrix, Vector<double> &solution, Vector<double> &system_rhs,
                      const FE_Q<dimension> &fe, const DoFHandler<dimension> &dof_handler)
{
  QGauss<dimension>  quadrature_formula(degree_finite_element+1);
  FEValues<dimension> fe_values (fe, quadrature_formula,
                         update_quadrature_points | update_values | update_gradients | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  Coeff<dimension> coeff;

  DoFHandler<dimension>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);

      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
	{
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
	      cell_matrix(i,j) += (coeff.value(fe_values.quadrature_point(q_index)) *
                                   fe_values.shape_grad (i, q_index) *
	        		   fe_values.shape_grad (j, q_index) *
	        		   fe_values.JxW (q_index));
            }
	  // for (unsigned int i=0; i<dofs_per_cell; ++i)
	    // cell_rhs(i) += (fe_values.shape_value (i, q_index) *
			    // 1 *
			    // fe_values.JxW (q_index));
	}
      cell->get_dof_indices (local_dof_indices);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));

      // for (unsigned int i=0; i<dofs_per_cell; ++i)
        // system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }


  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            ZeroFunction<dimension>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);

}



void run ()
{
    Triangulation<dimension>               triangulation;
    FE_Q<dimension>                        fe(degree_finite_element);
    DoFHandler<dimension>                  dof_handler(triangulation);
    ConstraintMatrix                 constraints;

    LaplaceOperatorGpu<dimension,degree_finite_element,double> system_matrix_gpu;
    LaplaceOperator<dimension,degree_finite_element,double> system_matrix;

    SparseMatrix<double> system_matrix_sparse;

    GridGenerator::hyper_cube (triangulation, 0., 1.);
    triangulation.refine_global (1);

    system_matrix.clear();
    system_matrix_gpu.clear();
    system_matrix_sparse.clear();

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    constraints.clear();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dimension>(),
                                              constraints);
    constraints.close();

    system_matrix.reinit (dof_handler, constraints);
    system_matrix_gpu.reinit (dof_handler, constraints);

    SparsityPattern      sparsity_pattern;
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, c_sparsity);
    sparsity_pattern.copy_from(c_sparsity);


    system_matrix_sparse.reinit (sparsity_pattern);

    Vector<double> vec(dof_handler.n_dofs());
    Vector<double> vec2(dof_handler.n_dofs());
    Vector<double> vec3(dof_handler.n_dofs());


    vec = 1.0;


    // fill in boundary values;
    std::vector<bool> bdry_ind;
    DoFTools::extract_boundary_dofs(dof_handler,ComponentMask(),bdry_ind);



    vec2 = 0.0;

    assemble_matrix(system_matrix_sparse,vec,vec2,fe,dof_handler);

    vec3 = vec2;

    GpuVector<double>                   gvec(vec);
    GpuVector<double>                   gvec2(vec2);

    // gvec = 1.0;
    // gvec2 = 0.0;

    // QGauss<dimension>  quadrature_formula(fe.degree+1);
    // FEValues<dimension> fe_values (fe, quadrature_formula,
                             // update_values   | update_gradients |
                             // update_quadrature_points | update_JxW_values );

    // const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    // const unsigned int   n_q_points    = quadrature_formula.size();

    // std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    // constraints.condense(system_rhs);

    // do stuff
    Timer time;

    time.reset();
    time.start();
    // for(int i = 0; i < Nreps; ++i) {
        system_matrix_gpu.vmult(gvec2,gvec);
        system_matrix.vmult(vec3,vec);
        system_matrix_sparse.vmult(vec2,vec);
    // }

    std::cout << "Time solve (CPU/wall) " << time() << "s/"
              << time.wall_time() << "s\n";


    Vector<double> gvec2_host = gvec2.toVector();
    std::cout << "spmv:" << std::endl;
    vec2.print("%8.4g");

    std::cout << "mfree_cpu: " << std::endl;
    vec3.print("%8.4g");

    std::cout << "mfree_gpu: " << std::endl;
    gvec2_host.print("%8.4g");


    system_matrix.clear();
    system_matrix_sparse.clear();
}




int main ()
{
    try
    {
        deallog.depth_console(0);
        run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
