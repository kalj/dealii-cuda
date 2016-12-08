#include "poisson_common.h"

using namespace dealii;

template <>
const Point<1>
Solution<1>::source_centers[Solution<1>::n_source_centers]
= { Point<1>(-1.0 / 3.0),
    Point<1>(0.0),
    Point<1>(+1.0 / 3.0)   };

template <>
const Point<2>
Solution<2>::source_centers[Solution<2>::n_source_centers]
= { Point<2>(-0.5, +0.5),
    Point<2>(-0.5, -0.5),
    Point<2>(+0.5, -0.5)   };

template <>
const Point<3>
Solution<3>::source_centers[Solution<3>::n_source_centers]
= { Point<3>(-0.5, +0.5, 0.25),
    Point<3>(-0.6, -0.5, -0.125),
    Point<3>(+0.5, -0.5, 0.5)   };

template <int dim>
const double
Solution<dim>::width = 1./3.;


template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];
    return_value += std::exp(-x_minus_xi.norm_square() /
                             (this->width * this->width));
  }

  return return_value /
    Utilities::fixed_power<dim>(std::sqrt(2 * pi) * this->width);
}



template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
  const double pi = numbers::PI;
  Tensor<1,dim> return_value;

  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

    return_value += (-2 / (this->width * this->width) *
                     std::exp(-x_minus_xi.norm_square() /
                              (this->width * this->width)) *
                     x_minus_xi);
  }

  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
                                                    this->width);
}

template <int dim>
double Solution<dim>::laplacian (const Point<dim>   &p,
                                 const unsigned int) const
{
  const double pi = numbers::PI;
  double return_value = 0;
  for (unsigned int i=0; i<this->n_source_centers; ++i)
  {
    const Tensor<1,dim> x_minus_xi = p - this->source_centers[i];

    double laplacian =
      ((-2*dim + 4*x_minus_xi.norm_square()/
        (this->width * this->width)) /
       (this->width * this->width) *
       std::exp(-x_minus_xi.norm_square() /
                (this->width * this->width)));
    return_value += laplacian;
  }
  return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * pi) *
                                                    this->width);
}


template class Solution<1> ;
template class Solution<2> ;
template class Solution<3> ;
