#include <deal.II/base/quadrature_lib.h>
#include <deal.II/matrix_free/fe_evaluation.h>


#include "poisson_common.h"
#include "laplace_operator_cpu.h"

template <int dim, int fe_degree, typename number>
LaplaceOperatorCpu<dim,fe_degree,number>::LaplaceOperatorCpu ()
  :
  Subscriptor()
{
  inverse_diagonal_matrix = std::make_shared<DiagonalMatrix<VectorType>>();
}



template <int dim, int fe_degree, typename number>
unsigned int
LaplaceOperatorCpu<dim,fe_degree,number>::m () const
{
  return data.get_vector_partitioner()->size();
}



template <int dim, int fe_degree, typename number>
unsigned int
LaplaceOperatorCpu<dim,fe_degree,number>::n () const
{
  return data.get_vector_partitioner()->size();
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::clear ()
{
  data.clear();
  diagonal_is_available = false;
  inverse_diagonal_matrix->clear();
}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::reinit (const DoFHandler<dim>  &dof_handler,
                                               const ConstraintMatrix  &constraints,
                                               const unsigned int      level)
{
  typename MatrixFree<dim,number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim,number>::AdditionalData::partition_color;
  additional_data.level_mg_handler = level;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);
  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);
  evaluate_coefficient();
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::
evaluate_coefficient ()
{
  const unsigned int n_cells = data.n_macro_cells();
  FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);
  coefficient.reinit (n_cells, phi.n_q_points);
  for (unsigned int cell=0; cell<n_cells; ++cell)
  {
    phi.reinit (cell);
    for (unsigned int q=0; q<phi.n_q_points; ++q)
      coefficient(cell,q) =
        Coefficient<dim>::value(phi.quadrature_point(q));
  }
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::
local_apply (const MatrixFree<dim,number>         &data,
             VectorType                       &dst,
             const VectorType                 &src,
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
LaplaceOperatorCpu<dim,fe_degree,number>::vmult (VectorType       &dst,
                                              const VectorType &src) const
{
  dst = 0;
  vmult_add (dst, src);
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::Tvmult (VectorType       &dst,
                                               const VectorType &src) const
{
  dst = 0;
  vmult_add (dst,src);
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::Tvmult_add (VectorType       &dst,
                                                      const VectorType &src) const
{
  vmult_add (dst,src);
}


template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::vmult_add (VectorType       &dst,
                                                  const VectorType &src) const
{
  data.cell_loop (&LaplaceOperatorCpu::local_apply, this, dst, src);

  const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i) {
    dst(constrained_dofs[i]) += src(constrained_dofs[i]);
    // dst.local_element(constrained_dofs[i]) += src.local_element(constrained_dofs[i]);
  }
}



template <int dim, int fe_degree, typename number>
number
LaplaceOperatorCpu<dim,fe_degree,number>::el (const unsigned int row,
                                           const unsigned int col) const
{
  Assert (row == col, ExcNotImplemented());
  Assert (diagonal_is_available == true, ExcNotInitialized());
  return 1.0/inverse_diagonal_matrix->operator()(row,col);
}


template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::set_diagonal(const VectorType &diagonal)
{
  AssertDimension (m(), diagonal.size());

  VectorType &diag = inverse_diagonal_matrix->get_vector();

  diag.reinit(m());
  for(int i = 0; i < m(); ++i) {
    diag[i] = 1.0/diagonal[i];
  }

  const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    diag(constrained_dofs[i]) = 1.0;

  diagonal_is_available = true;
}

template <int dim, int fe_degree, typename number>
const std::shared_ptr<DiagonalMatrix<typename LaplaceOperatorCpu<dim,fe_degree,number>::VectorType>>
LaplaceOperatorCpu<dim,fe_degree,number>::get_diagonal_inverse() const
{
  Assert (diagonal_is_available == true, ExcNotInitialized());
  return inverse_diagonal_matrix;
}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::initialize_dof_vector(VectorType &vector) const
{
  // if (!vector.partitioners_are_compatible(*data.get_dof_info(0).vector_partitioner))
    // data.initialize_dof_vector(vector);
    data.initialize_dof_vector(vector);
  // Assert(vector.partitioners_are_globally_compatible(*data.get_dof_info(0).vector_partitioner),
         // ExcInternalError());
}



template <int dim, int fe_degree, typename number>
std::size_t
LaplaceOperatorCpu<dim,fe_degree,number>::memory_consumption () const
{
  return (data.memory_consumption () +
          coefficient.memory_consumption() +
          MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
          MemoryConsumption::memory_consumption(diagonal_is_available));
}

template class LaplaceOperatorCpu<2,1,double>;
template class LaplaceOperatorCpu<2,2,double>;
template class LaplaceOperatorCpu<2,3,double>;
template class LaplaceOperatorCpu<2,4,double>;

template class LaplaceOperatorCpu<3,1,double>;
template class LaplaceOperatorCpu<3,2,double>;
template class LaplaceOperatorCpu<3,3,double>;
template class LaplaceOperatorCpu<3,4,double>;
