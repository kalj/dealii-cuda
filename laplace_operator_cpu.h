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

  LaplaceOperatorCpu ();

  void clear();

  void reinit (const DoFHandler<dim>  &dof_handler,
               const ConstraintMatrix  &constraints,
               const unsigned int      level = numbers::invalid_unsigned_int);

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

  number el (const unsigned int row,
             const unsigned int col) const;


  void
  initialize_dof_vector(VectorType &vector) const;

  // diagonal for preconditioning
  void set_diagonal (const VectorType &diagonal);

  const std::shared_ptr<DiagonalMatrix<VectorType>> get_diagonal_inverse () const;

  std::size_t memory_consumption () const;

private:
  void local_apply (const MatrixFree<dim,number>    &data,
                    VectorType                      &dst,
                    const VectorType                &src,
                    const std::pair<unsigned int,unsigned int> &cell_range) const;

  void evaluate_coefficient();

  MatrixFree<dim,number>      data;
  Table<2, VectorizedArray<number> > coefficient;

  std::shared_ptr<DiagonalMatrix<VectorType>>  inverse_diagonal_matrix;
  bool            diagonal_is_available;
};


#endif /* _LAPLACE_OPERATOR_CPU_H */
