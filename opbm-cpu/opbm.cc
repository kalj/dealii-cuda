/* ---------------------------------------------------------------------
 * $Id$
 *
 * Copyright (C) 2009 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University, 2009-2012
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>


#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <fstream>
#include <sstream>


namespace Step37
{
  using namespace dealii;

#define N_ITERATIONS 100

#ifdef DEGREE_FE
const unsigned int degree_finite_element = DEGREE_FE;
#else
const unsigned int degree_finite_element = 3;
#endif

#ifdef DIMENSION
const unsigned int dimension = DIMENSION;
#else
const unsigned int dimension = 2;
#endif


  template <int dim>
  class Coefficient : public Function<dim>
  {
  public:
    Coefficient ()  : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    template <typename number>
    number value (const Point<dim,number> &p,
                  const unsigned int component = 0) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;
  };



  template <int dim>
  template <typename number>
  number Coefficient<dim>::value (const Point<dim,number> &p,
                                  const unsigned int /*component*/) const
  {
    return 1. / (0.05 + 2.*p.square());
  }



  template <int dim>
  double Coefficient<dim>::value (const Point<dim>  &p,
                                  const unsigned int component) const
  {
    return value<double>(p,component);
  }



  template <int dim>
  void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double>            &values,
                                     const unsigned int              component) const
  {
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    const unsigned int n_points = points.size();
    for (unsigned int i=0; i<n_points; ++i)
      values[i] = value<double>(points[i],component);
  }





  template <int dim, int fe_degree, typename number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    LaplaceOperator ();

    void clear();

    void reinit (const DoFHandler<dim>  &dof_handler,
                 const ConstraintMatrix  &constraints
                 );

    unsigned int m () const;
    unsigned int n () const;

    void vmult (Vector<double> &dst,
                const Vector<double> &src) const;
    void Tvmult (Vector<double> &dst,
                 const Vector<double> &src) const;
    void vmult_add (Vector<double> &dst,
                    const Vector<double> &src) const;
    void Tvmult_add (Vector<double> &dst,
                     const Vector<double> &src) const;

    number el (const unsigned int row,
               const unsigned int col) const;
    void set_diagonal (const Vector<number> &diagonal);

    const Vector<number>& get_diagonal () const {
      Assert (diagonal_is_available == true, ExcNotInitialized());
      return diagonal_values;
    };

    std::size_t memory_consumption () const;

  private:
    void local_apply (const MatrixFree<dim,number>    &data,
                      Vector<double>                      &dst,
                      const Vector<double>                &src,
                      const std::pair<unsigned int,unsigned int> &cell_range) const;

    void evaluate_coefficient(const Coefficient<dim> &function);

    MatrixFree<dim,number>      data;
    Table<2, VectorizedArray<number> > coefficient;

    Vector<number>  diagonal_values;
    bool            diagonal_is_available;
  };



  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim,fe_degree,number>::LaplaceOperator ()
    :
    Subscriptor()
  {}



  template <int dim, int fe_degree, typename number>
  unsigned int
  LaplaceOperator<dim,fe_degree,number>::m () const
  {
    return data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, typename number>
  unsigned int
  LaplaceOperator<dim,fe_degree,number>::n () const
  {
    return data.get_vector_partitioner()->size();
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::clear ()
  {
    data.clear();
    diagonal_is_available = false;
    diagonal_values.reinit(0);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::reinit (const DoFHandler<dim>  &dof_handler,
                                                 const ConstraintMatrix  &constraints)
  {
    typename MatrixFree<dim,number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,number>::AdditionalData::partition_color;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points);
    data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
                 additional_data);
    evaluate_coefficient(Coefficient<dim>());
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  evaluate_coefficient (const Coefficient<dim> &coefficient_function)
  {
    const unsigned int n_cells = data.n_macro_cells();
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
    coefficient.reinit (n_cells, phi.n_q_points);
    for (unsigned int cell=0; cell<n_cells; ++cell)
      {
        phi.reinit (cell);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          coefficient(cell,q) =
            coefficient_function.value(phi.quadrature_point(q));
      }
  }




  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  local_apply (const MatrixFree<dim,number>         &data,
               Vector<double>                       &dst,
               const Vector<double>                 &src,
               const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false,true,false);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          phi.submit_gradient (coefficient(cell,q) *
                               phi.get_gradient(q), q);
        phi.integrate (false,true);
        phi.distribute_local_to_global (dst);
      }
  }




  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::vmult (Vector<double>       &dst,
                                                const Vector<double> &src) const
  {
    dst = 0;
    vmult_add (dst, src);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::Tvmult (Vector<double>       &dst,
                                                 const Vector<double> &src) const
  {
    dst = 0;
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::Tvmult_add (Vector<double>       &dst,
                                                     const Vector<double> &src) const
  {
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::vmult_add (Vector<double>       &dst,
                                                    const Vector<double> &src) const
  {
    data.cell_loop (&LaplaceOperator::local_apply, this, dst, src);

    const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      dst(constrained_dofs[i]) += src(constrained_dofs[i]);
  }



  template <int dim, int fe_degree, typename number>
  number
  LaplaceOperator<dim,fe_degree,number>::el (const unsigned int row,
                                             const unsigned int col) const
  {
    Assert (row == col, ExcNotImplemented());
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return diagonal_values(row);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::set_diagonal(const Vector<number> &diagonal)
  {
    AssertDimension (m(), diagonal.size());

    diagonal_values = diagonal;

    const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
    for (unsigned int i=0; i<constrained_dofs.size(); ++i)
      diagonal_values(constrained_dofs[i]) = 1.0;

    diagonal_is_available = true;
  }



  template <int dim, int fe_degree, typename number>
  std::size_t
  LaplaceOperator<dim,fe_degree,number>::memory_consumption () const
  {
    return (data.memory_consumption () +
            coefficient.memory_consumption() +
            diagonal_values.memory_consumption() +
            MemoryConsumption::memory_consumption(diagonal_is_available));
  }


  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run ();

  private:
    void setup_system ();
    void solve ();

    typedef LaplaceOperator<dim,degree_finite_element,double> SystemMatrixType;

    Triangulation<dim>               triangulation;
    FE_Q<dim>                        fe;
    DoFHandler<dim>                  dof_handler;
    ConstraintMatrix                 constraints;

    SystemMatrixType                 system_matrix;

    Vector<double>                   src;
    Vector<double>                   dst;

    double                           setup_time;
    ConditionalOStream               time_details;
    unsigned int                     n_iterations;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    fe (degree_finite_element),
    dof_handler (triangulation),
    time_details (std::cout, false),
    n_iterations(N_ITERATIONS)
  {}




  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    Timer time;
    time.start ();
    setup_time = 0;

    system_matrix.clear();

    dof_handler.distribute_dofs (fe);

    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;


  std::cout << "Number of elements: "
            << dof_handler.get_tria().n_active_cells()
            << std::endl;

    constraints.clear();
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close();
    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) "
                 << time() << "s/" << time.wall_time() << "s" << std::endl;
    time.restart();

    system_matrix.reinit (dof_handler, constraints);
    std::cout.precision(4);
    std::cout << "System matrix memory consumption:     "
              << system_matrix.memory_consumption()*1e-6
              << " MB."
              << std::endl;

    dst.reinit (system_matrix.n());
    src.reinit (system_matrix.n());

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) "
                 << time() << "s/" << time.wall_time() << "s" << std::endl;
    time.restart();

  }


  template <int dim>
  void LaplaceProblem<dim>::solve ()
  {
    Timer time;

    // IC
    dst = 0.1;

    for(unsigned int i = 0; i < n_iterations; ++i) {
      dst.swap(src);

      system_matrix.vmult(dst,src);
    }

    time.stop();

    std::cout << "Time solve ("
              << n_iterations
              << " iterations)  (CPU/wall) " << time() << "s/"
              << time.wall_time() << "s\n";

    std::cout << "Per iteration "
              << time.wall_time() / n_iterations << "s\n";
  }




  template <int dim>
  void LaplaceProblem<dim>::run ()
  {

    GridGenerator::hyper_cube (triangulation, 0., 1.);

      // triangulation.refine_global (1);

    if(dim == 2) {
      triangulation.refine_global (6);
    }
    else if(dim == 3) {
      triangulation.refine_global (4);
    }

    if(degree_finite_element ==1) {
      triangulation.refine_global (3);
    }
    else if(degree_finite_element ==2) {
      triangulation.refine_global (2);
    }
    else if(degree_finite_element ==3) {
      triangulation.refine_global (1);
    }
    else if(degree_finite_element ==4) {
      triangulation.refine_global (1);

    }

    setup_system ();
    solve ();
    std::cout << std::endl;
  }
}




int main ()
{
  try
    {
      using namespace Step37;

      deallog.depth_console(0);
       printf("d: %d, p: %d\n",dimension,degree_finite_element);
      LaplaceProblem<dimension> laplace_problem;
      laplace_problem.run ();
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
