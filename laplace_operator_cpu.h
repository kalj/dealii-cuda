/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#ifndef _LAPLACE_OPERATOR_CPU_H
#define _LAPLACE_OPERATOR_CPU_H

#include <deal.II/base/subscriptor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/diagonal_matrix.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>


using namespace dealii;



//=============================================================================
// operator
//=============================================================================

template <int dim, int fe_degree, typename number>
class LaplaceOperatorCpu : public Subscriptor
{
public:
  // typedef LinearAlgebra::distributed::Vector<number> VectorType;
  typedef Vector<number> VectorType;
  typedef number         value_type;

  LaplaceOperatorCpu ();

  void clear();

  void reinit (const DoFHandler<dim>   &dof_handler,
               const ConstraintMatrix  &constraints);


  void reinit (const DoFHandler<dim>    &dof_handler,
               const MGConstrainedDoFs  &mg_constrained_dofs,
               const unsigned int        level);

  unsigned int m () const;
  unsigned int n () const;

  void vmult (VectorType &dst,
              const VectorType &src) const;
  void Tvmult (VectorType &dst,
               const VectorType &src) const;
  void vmult_add (VectorType &dst,
                  const VectorType &src) const;
  void Tvmult_add (VectorType &dst,
                   const VectorType &src) const;
  void vmult_interface_down(VectorType       &dst,
                            const VectorType &src) const;
  void vmult_interface_up(VectorType       &dst,
                          const VectorType &src) const;

  number el (const unsigned int row,
             const unsigned int col) const;


  void
  initialize_dof_vector(VectorType &vector) const;

  const std::shared_ptr<DiagonalMatrix<VectorType>> get_diagonal_inverse () const;

  void compute_diagonal();

  std::size_t memory_consumption () const;

private:
  std::vector<unsigned int> edge_constrained_indices;
  mutable std::vector<std::pair<number,number> > edge_constrained_values;

  void local_apply (const MatrixFree<dim,number>    &data,
                    VectorType                      &dst,
                    const VectorType                &src,
                    const std::pair<unsigned int,unsigned int> &cell_range) const;

  void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                               VectorType       &dst,
                               const unsigned int                               &dummy,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const;

  void evaluate_coefficient();

  MatrixFree<dim,number>      data;
  Table<2, VectorizedArray<number> > coefficient;

  std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
  bool            diagonal_is_available;
};


#endif /* _LAPLACE_OPERATOR_CPU_H */
