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
                                                  const ConstraintMatrix  &constraints)
{
  typename MatrixFree<dim,number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim,number>::AdditionalData::partition_color;
  additional_data.level_mg_handler = numbers::invalid_unsigned_int;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);

  data.reinit (dof_handler, constraints, QGauss<1>(fe_degree+1),
               additional_data);
  evaluate_coefficient();

  // edge constraints are handled by MatrixFree as hanging-node constraints
  edge_constrained_indices.clear();
  edge_constrained_values.clear();

}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::reinit (const DoFHandler<dim>    &dof_handler,
                                                  const MGConstrainedDoFs  &mg_constrained_dofs,
                                                  const unsigned int        level)
{
  // only pass boundary constraints to MatrixFree::reinit, refinement edge
  // constraints are handled below
  ConstraintMatrix level_constraints;
  level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
  level_constraints.close();

  typename MatrixFree<dim,number>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    MatrixFree<dim,number>::AdditionalData::partition_color;
  additional_data.level_mg_handler = level;
  additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                          update_quadrature_points);

  data.reinit (dof_handler, level_constraints, QGauss<1>(fe_degree+1),
               additional_data);
  evaluate_coefficient();


  edge_constrained_indices.clear();
  edge_constrained_values.clear();

  std::vector<types::global_dof_index> interface_indices;
  mg_constrained_dofs.get_refinement_edge_indices(level).fill_index_vector(interface_indices);
  edge_constrained_indices = interface_indices;
  edge_constrained_values.resize(interface_indices.size());

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
  // For MG, we need to take care of internal refinement edge constraints.
  // This means that edge DoFs are treated as homogeneous Dirichlet boundary conditions.
  // To do this, we temporarily set them to 0, and then restore them afterwards.
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    edge_constrained_values[i] =
      std::pair<number,number>(src(edge_constrained_indices[i]),
                               dst(edge_constrained_indices[i]));
    const_cast<VectorType&>(src)(edge_constrained_indices[i]) = 0.;
  }


  data.cell_loop (&LaplaceOperatorCpu::local_apply, this, dst, src);

  // copy the real boundary Dirichlet values, which are untouched
  const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i) {
    dst(constrained_dofs[i]) += src(constrained_dofs[i]);
  }

  // reset edge constrained values, multiply by unit matrix and add into
  // destination
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    const_cast<VectorType&>(src)(edge_constrained_indices[i]) = edge_constrained_values[i].first;
    dst(edge_constrained_indices[i]) = edge_constrained_values[i].second + edge_constrained_values[i].first;
  }
}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::
vmult_interface_down(VectorType       &dst,
                     const VectorType &src) const
{
  dst = 0;

  // set zero Dirichlet values on the refinement edges of the input vector (and
  // remember the src and dst values because we need to reset them at the end)
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    edge_constrained_values[i] =
      std::pair<number,number>(src(edge_constrained_indices[i]),
                               dst(edge_constrained_indices[i]));
    const_cast<VectorType&>(src)(edge_constrained_indices[i]) = 0.;
  }

  data.cell_loop (&LaplaceOperatorCpu::local_apply, this, dst, src);


  // now zero out everything except the values at the refinement edges, and
  // restore the src values
  unsigned int c=0;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    for ( ; c<edge_constrained_indices[i]; ++c)
      dst(c) = 0.;
    ++c;

    // reset the src values
    const_cast<VectorType&>(src)(edge_constrained_indices[i]) = edge_constrained_values[i].first;
  }
  for ( ; c<dst.size(); ++c)
    dst(c) = 0.;
}

template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>::
vmult_interface_up(VectorType       &dst,
                   const VectorType &src) const
{
  dst = 0;

  VectorType src_cpy (src);
  unsigned int c=0;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    for ( ; c<edge_constrained_indices[i]; ++c)
      src_cpy(c) = 0.;
    ++c;
  }
  for ( ; c<src_cpy.size(); ++c)
    src_cpy(c) = 0.;

  data.cell_loop (&LaplaceOperatorCpu::local_apply, this, dst, src_cpy);

  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
  {
    dst(edge_constrained_indices[i]) = 0.;
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
LaplaceOperatorCpu<dim,fe_degree,number>::compute_diagonal()
{
  inverse_diagonal_matrix.
    reset(new DiagonalMatrix<VectorType >());
  VectorType &inverse_diagonal = inverse_diagonal_matrix->get_vector();

  data.initialize_dof_vector(inverse_diagonal);

  unsigned int dummy = 0;
  data.cell_loop (&LaplaceOperatorCpu::local_compute_diagonal, this,
                  inverse_diagonal, dummy);

  const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    inverse_diagonal(constrained_dofs[i]) = 1.;
  for (unsigned int i=0; i<edge_constrained_indices.size(); ++i)
    inverse_diagonal(edge_constrained_indices[i]) = 1.;

  for (unsigned int i=0; i<inverse_diagonal.size(); ++i)
    inverse_diagonal(i) =
      1./inverse_diagonal(i);
}




template <int dim, int fe_degree, typename number>
void
LaplaceOperatorCpu<dim,fe_degree,number>
::local_compute_diagonal (const MatrixFree<dim,number>               &data,
                          VectorType &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int> &cell_range) const
{
  FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

  AlignedVector<VectorizedArray<number> > diagonal(phi.dofs_per_cell);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    phi.reinit (cell);
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
    {
      for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
        phi.submit_dof_value(VectorizedArray<number>(), j);
      phi.submit_dof_value(make_vectorized_array<number>(1.), i);

      phi.evaluate (false, true);
      for (unsigned int q=0; q<phi.n_q_points; ++q)
        phi.submit_gradient (coefficient(cell,q) *
                             phi.get_gradient(q), q);
      phi.integrate (false, true);
      diagonal[i] = phi.get_dof_value(i);
    }
    for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
      phi.submit_dof_value(diagonal[i], i);
    phi.distribute_local_to_global (dst);
  }
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
