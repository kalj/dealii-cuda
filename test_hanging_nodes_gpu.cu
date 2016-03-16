/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)test_hanging_nodes_gpu.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#include <cstdio>
#include <iostream>
#include <fstream>


#include <deal.II/lac/vector.h>

// #include <deal.II/fe/fe.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/base/quadrature_lib.h>

// #include <deal.II/lac/vector.templates.h>
// #include "/local/home/karll/build/dealii/source/numerics/data_out_dof_data.cc"
// #include "/local/home/karll/build/dealii/source/fe/fe_values.cc"


#include "matrix_free_gpu/defs.h"
// #include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/matrix_free_gpu.h"
// #include "matrix_free_gpu/fee_gpu.cuh"
// #include "matrix_free_gpu/cuda_utils.cuh"

using namespace dealii;

template <int dim>
void write_mesh(const DoFHandler<dim> &dof_handler,
                const char *fname)
{
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.build_patches ();
    std::ofstream output(fname);
    data_out.write_vtu(output);
}

int main(int argc, char *argv[])
{

  typedef double Number;
  const unsigned int dimension = 3;
  const unsigned int fe_degree = 1;

  Triangulation<dimension>               triangulation;
  FE_Q<dimension>                        fe(fe_degree);
  DoFHandler<dimension>                  dof_handler(triangulation);
  Quadrature<dimension> q(fe.get_unit_support_points());
  FEValues<dimension> fe_values(fe,q,update_q_points);

  enum test_cases { cube_corner, slab_edge, loop, twisted_loop, moebius};

  test_cases test_case = slab_edge;

  //---------------------------------------------------------------------------
  // Setup basic mesh
  //---------------------------------------------------------------------------

  switch(test_case) {
  case cube_corner:
    GridGenerator::hyper_cube (triangulation, 0., 1.);
    triangulation.refine_global (1);
    break;
  case slab_edge:
    {
      std::vector<unsigned int> reps;
      reps.push_back(2);
      reps.push_back(1);
      reps.push_back(2);
      GridGenerator::subdivided_hyper_rectangle (triangulation,
                                                 reps,Point<3>(0.0,0.0,0.0),
                                                 Point<3>(1.0,0.5,1.0));
    }
    break;
  case loop:
    {
      std::vector<Point<3> > vertices;
      vertices.push_back(Point<3>(0,  0,0));
      vertices.push_back(Point<3>(1,  0,0));
      vertices.push_back(Point<3>(1,  1,0));
      vertices.push_back(Point<3>(2.2,0.2,0));
      vertices.push_back(Point<3>(1.8,1.1,0));
      vertices.push_back(Point<3>(3,  1.6,0));
      vertices.push_back(Point<3>(2.1,1.6,0));
      vertices.push_back(Point<3>(2.6,2.6,0));
      vertices.push_back(Point<3>(2,  2,0));
      vertices.push_back(Point<3>(1.6,2.1,0));
      vertices.push_back(Point<3>(1.6,3,0));

      vertices.push_back(Point<3>(1.1,1.8,0));
      vertices.push_back(Point<3>(0.2,2.2,0));
      vertices.push_back(Point<3>(0,  1,0));

      vertices.push_back(Point<3>(0,  0,1));
      vertices.push_back(Point<3>(1,  0,1));
      vertices.push_back(Point<3>(1,  1,1));
      vertices.push_back(Point<3>(2.2,0.2,1));
      vertices.push_back(Point<3>(1.8,1.1,1));
      vertices.push_back(Point<3>(3,  1.6,1));
      vertices.push_back(Point<3>(2.1,1.6,1));
      vertices.push_back(Point<3>(2.6,2.6,1));
      vertices.push_back(Point<3>(2,  2,1));
      vertices.push_back(Point<3>(1.6,2.1,1));
      vertices.push_back(Point<3>(1.6,3,1));

      vertices.push_back(Point<3>(1.1,1.8,1));
      vertices.push_back(Point<3>(0.2,2.2,1));

      vertices.push_back(Point<3>(0,  1,1));

      std::vector<CellData<3> > cells(7,CellData<3>());
      cells[0].vertices[0] = 0;
      cells[0].vertices[1] = 1;
      cells[0].vertices[2] = 13;
      cells[0].vertices[3] = 2;
      cells[0].material_id = 0;

      cells[1].vertices[0] = 1;
      cells[1].vertices[1] = 3;
      cells[1].vertices[2] = 2;
      cells[1].vertices[3] = 4;
      cells[1].material_id = 0;

      cells[2].vertices[0] = 3;
      cells[2].vertices[1] = 5;
      cells[2].vertices[2] = 4;
      cells[2].vertices[3] = 6;
      cells[2].material_id = 0;

      cells[3].vertices[0] = 5;
      cells[3].vertices[1] = 7;
      cells[3].vertices[2] = 6;
      cells[3].vertices[3] = 8;
      cells[3].material_id = 0;

      cells[4].vertices[0] = 7;
      cells[4].vertices[1] = 10;
      cells[4].vertices[2] = 8;
      cells[4].vertices[3] = 9;
      cells[4].material_id = 0;

      cells[5].vertices[0] = 10;
      cells[5].vertices[1] = 12;
      cells[5].vertices[2] = 9;
      cells[5].vertices[3] = 11;
      cells[5].material_id = 0;

      cells[6].vertices[0] = 12;
      cells[6].vertices[1] = 13;
      cells[6].vertices[2] = 11;
      cells[6].vertices[3] = 2;
      cells[6].material_id = 0;

      for(int c = 0; c < 7; ++c) {
        for(int i = 0; i < 4; ++i)
          cells[c].vertices[4+i] = cells[c].vertices[i]+14;
      }

      triangulation.create_triangulation(vertices,cells,SubCellData());


      break;
    }
  case twisted_loop:
    {
      float diag = sqrt(2)/2;
      float ang = 0.1;
      float rot = 0.6;
      float pi = 3.1415926535;

      std::vector<Point<3> > vertices;
      vertices.push_back(Point<3>(0,  0,0));
      vertices.push_back(Point<3>(1,  0,0));
      vertices.push_back(Point<3>(1,  1,0));
      vertices.push_back(Point<3>(2.2,0.2,0));
      vertices.push_back(Point<3>(1.8,1.1,0));
      vertices.push_back(Point<3>(3,  1.6,0));
      vertices.push_back(Point<3>(2.1,1.6,0));
      vertices.push_back(Point<3>(2.6,2.6,0));
      vertices.push_back(Point<3>(2,  2,0));
      vertices.push_back(Point<3>(1.6,2.1,0));
      vertices.push_back(Point<3>(1.6,3,0));

      vertices.push_back(Point<3>(0.65+diag*cos(pi*ang)*cos(pi*(1.5+rot)),2-diag*sin(pi*ang)*cos(pi*(1.5+rot)),0.5+diag*sin((1.5+rot)*pi)));
      vertices.push_back(Point<3>(0.65+diag*cos(pi*ang)*cos(pi*(1+rot)),2-diag*sin(pi*ang)*cos(pi*(1+rot)),0.5+diag*sin((1+rot)*pi)));

      vertices.push_back(Point<3>(0,  1,0));

      vertices.push_back(Point<3>(0,  0,1));
      vertices.push_back(Point<3>(1,  0,1));
      vertices.push_back(Point<3>(1,  1,1));
      vertices.push_back(Point<3>(2.2,0.2,1));
      vertices.push_back(Point<3>(1.8,1.1,1));
      vertices.push_back(Point<3>(3,  1.6,1));
      vertices.push_back(Point<3>(2.1,1.6,1));
      vertices.push_back(Point<3>(2.6,2.6,1));
      vertices.push_back(Point<3>(2,  2,1));
      vertices.push_back(Point<3>(1.6,2.1,1));
      vertices.push_back(Point<3>(1.6,3,1));

      vertices.push_back(Point<3>(0.65+diag*cos(pi*ang)*cos(pi*rot),2-diag*sin(pi*ang)*cos(pi*rot),0.5+diag*sin(rot*pi)));
      vertices.push_back(Point<3>(0.65+diag*cos(pi*ang)*cos(pi*(0.5+rot)),2-diag*sin(pi*ang)*cos(pi*(0.5+rot)),0.5+diag*sin((0.5+rot)*pi)));

      vertices.push_back(Point<3>(0,  1,1));

      std::vector<CellData<3> > cells(7,CellData<3>());
      cells[0].vertices[0] = 0;
      cells[0].vertices[1] = 1;
      cells[0].vertices[2] = 13;
      cells[0].vertices[3] = 2;
      cells[0].material_id = 0;

      cells[1].vertices[0] = 1;
      cells[1].vertices[1] = 3;
      cells[1].vertices[2] = 2;
      cells[1].vertices[3] = 4;
      cells[1].material_id = 0;

      cells[2].vertices[0] = 3;
      cells[2].vertices[1] = 5;
      cells[2].vertices[2] = 4;
      cells[2].vertices[3] = 6;
      cells[2].material_id = 0;

      cells[3].vertices[0] = 5;
      cells[3].vertices[1] = 7;
      cells[3].vertices[2] = 6;
      cells[3].vertices[3] = 8;
      cells[3].material_id = 0;

      cells[4].vertices[0] = 7;
      cells[4].vertices[1] = 10;
      cells[4].vertices[2] = 8;
      cells[4].vertices[3] = 9;
      cells[4].material_id = 0;

      cells[5].vertices[0] = 10;
      cells[5].vertices[1] = 12;
      cells[5].vertices[2] = 9;
      cells[5].vertices[3] = 11;
      cells[5].material_id = 0;

      for(int c = 0; c < 6; ++c) {
        for(int i = 0; i < 4; ++i)
          cells[c].vertices[4+i] = cells[c].vertices[i]+14;
      }

      cells[6].vertices[0] = 12;
      cells[6].vertices[1] = 2;
      cells[6].vertices[2] = 11;
      cells[6].vertices[3] = 16;
      cells[6].vertices[4] = 26;
      cells[6].vertices[5] = 13;
      cells[6].vertices[6] = 25;
      cells[6].vertices[7] = 27;
      cells[6].material_id = 0;

      triangulation.create_triangulation(vertices,cells,SubCellData());

      break;
    }
  case moebius:
    GridGenerator::moebius (triangulation, 8, 1, 2.0, 1.0);
    break;
  default:
    std::cerr << "incorrect grid type!" << std::endl;
    exit(1);
  }

  dof_handler.distribute_dofs (fe);

  write_mesh<dimension> (dof_handler,"grid-0.vtu");

  //---------------------------------------------------------------------------
  // Setup refinement
  //---------------------------------------------------------------------------

  typename DoFHandler<dimension>::active_cell_iterator
    it = dof_handler.begin_active(),
    end = dof_handler.end();
  for(int cellid=0; it != end; ++it, cellid++) {
    Point<dimension> p = it->center();

    // std::cout << "cell " << cellid << ": " << p << std::endl;

    bool ref=false;
    if(test_case == moebius) {
      ref = p[0] > 1 && p[1] < 0;
    //ref = p[0] > 1 && p[1] > 0;
    }
    else if(test_case == cube_corner) {

      ref=true;
      for(int d = 0; d < dimension; ++d) {
        ref = ref && p[d] > 0.5;
      }
    }
    else if(test_case == slab_edge) {

      ref=p[0] > 0.5 || p[2] > 0.5;

    }
    else if(test_case == loop || test_case == twisted_loop) {
      ref = p[0] < 1 && p[1] > 1;
    }

    if(ref) it->set_refine_flag();
  }

  printf("Number of DoFs: %d\n",dof_handler.n_dofs());

  triangulation.execute_coarsening_and_refinement();

  dof_handler.distribute_dofs (fe);

  write_mesh<dimension> (dof_handler,"grid-1.vtu");

  //---------------------------------------------------------------------------
  // print properties of mesh
  //---------------------------------------------------------------------------

  it = dof_handler.begin_active();
  end = dof_handler.end();

  printf("\n");
  printf("new mesh:\n");
  printf("Number of DoFs: %d\n",dof_handler.n_dofs());
  for(int cellid=0; it != end; ++it, cellid++) {

    Point<dimension> p = it->center();
    if(true) {
    // if(p[0] > 1.5 && p[1] > -0.5) {
      // if(p[0] > 1.5 && p[1] < 0.5) {

      std::cout << "---------------------------" << std::endl;
      std::cout << "cell " << cellid << ":" << std::endl;
      std::cout << "---------------------------" << std::endl;
      std::cout << "  center: " << p << std::endl;
      std::cout << "  dofs:" << std::endl;
      std::vector<unsigned int> dof_indices(fe.dofs_per_cell);
      fe_values.reinit(it);

      const std::vector<Point<dimension> > &support_points = fe_values.get_quadrature_points();


      it->get_dof_indices(dof_indices);
      for(int i = 0; i < dof_indices.size(); ++i) {
        std::cout << "    " << dof_indices[i] << " : [" << support_points[i] << "]" << std::endl;
      }

      for (unsigned int face=0; face<GeometryInfo<dimension>::faces_per_cell; ++face) {

        const typename DoFHandler<dimension>::active_cell_iterator &neighbor = it->neighbor(face);
        if(!it->at_boundary(face)) {



          Point<dimension> fp = it->face(face)->center();

          bool apa = true;
          if(test_case == moebius) apa = (fp[1]==0 && fp[0] > 0 ) || cellid==0;
          else if(test_case == loop || test_case==twisted_loop) apa = fp[1] == 1;

          if(apa) {

            const unsigned int neighbor_face_index = it->neighbor_face_no(face);

            printf("face %d connected to face %d of neighbor\n",face, neighbor_face_index);

            if(neighbor->level() < it->level()) {
              unsigned int subface=0;
              for (; subface< GeometryInfo<dimension>::max_children_per_face; ++subface)
                if(neighbor->neighbor_child_on_subface(neighbor_face_index,subface) == it)
                  break;

              printf("  (subface %d)\n",subface);
            }

            std::cout << "face props: " << it->face_orientation(face) << ", "
                      << it->face_rotation(face) << ", " << it->face_flip(face) << std::endl;


            std::vector<types::global_dof_index> face_dofs(fe.dofs_per_face);
            it->face(face)->get_dof_indices(face_dofs);

            printf("dofs on face: ");
            for(int i = 0; i < fe.dofs_per_face; ++i) {
              printf("%8d",face_dofs[i]);
            }
            printf("\n");
          }
        }
      }
    }
  }


  /*
  ConstraintMatrix hn_constraints;
  DoFTools::make_hanging_node_constraints(dof_handler,hn_constraints);

  // VectorTools::interpolate_boundary_values(dof_handler,
  //                                          0,
  //                                          ZeroFunction<dimension>(),
  //                                          hn_constraints);

  hn_constraints.close();

  printf("\nConstrained DoFs:\n\n");
  for(int i = 0; i < dof_handler.n_dofs(); ++i) {
    if(hn_constraints.is_constrained(i)) {
      printf("DoF %d is constrained:\n",i);
      const std::vector<std::pair<unsigned int,double> > *entries = hn_constraints.get_constraint_entries(i);

      for(int j = 0; j < entries->size(); ++j) {
        std::pair<unsigned int,double> p = (*entries)[j];
        printf("  (%d,%g)\n",p.first,p.second);
      }
      if(hn_constraints.is_inhomogeneously_constrained(i))
        printf("  rhs=%g\n",hn_constraints.get_inhomogeneity(i));
    }
  }
  */


  ConstraintMatrix boundary_constraints;
  /*
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           ZeroFunction<dimension>(),
                                           boundary_constraints);
  */

  boundary_constraints.close();


  MatrixFreeGpu<dimension,Number> mf_data;

  typename MatrixFreeGpu<dimension,Number>::AdditionalData additional_data;
  additional_data.use_coloring = false;
  additional_data.parallelization_scheme = MatrixFreeGpu<dimension,Number>::scheme_par_in_elem;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);

  mf_data.reinit(dof_handler,boundary_constraints,QGauss<1>(fe_degree+1),additional_data);


  return 0;
}
