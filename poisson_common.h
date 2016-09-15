
#ifndef _POISSON_COMMON_H
#define _POISSON_COMMON_H

//=============================================================================
// reference solution and right-hand side
//=============================================================================

template <int dim>
class Solution : public Function<dim>
{
private:
  static const unsigned int n_source_centers = 3;
  static const Point<dim>   source_centers[n_source_centers];
  static const double       width;

public:
  Solution () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;

  virtual double laplacian (const Point<dim>   &p,
                            const unsigned int  component = 0) const;
};

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

//=============================================================================
// coefficient
//=============================================================================

template <int dim>
struct Coefficient
{
  static __host__ double value (const Point<dim> &p){
    return 1. / (0.05 + 2.*(p.norm_square()));
    // return 1.;
  }

  template <typename Number>
  static __device__ Number value (const GpuArray<dim,Number> &p){
    return 1. / (0.05 + 2.*(p.norm_square()));
    // return 1.;
  }

  static __host__  Tensor<1,dim> gradient (const Point<dim> &p){
    const Tensor<1,dim> dist = -p;
    const double den = 0.05 + 2.*dist.norm_square();
    return (4. / (den*den))*dist;
    // const Tensor<1,dim> dist = p*0;
    // return dist;
  }
};

// Wrapper for coefficient
template <int dim>
class CoefficientFun : Function<dim>
{
public:
  CoefficientFun () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;

  virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                  const unsigned int  component = 0) const;
};

template <int dim>
double CoefficientFun<dim>::value (const Point<dim>   &p,
                                   const unsigned int) const
{
  return Coefficient<dim>::value(p); // 1. / (0.05 + 2.*((p-x_c).norm_square()));
}

template <int dim>
void CoefficientFun<dim>::value_list (const std::vector<Point<dim> > &points,
                                      std::vector<double>            &values,
                                      const unsigned int              component) const
{
  Assert (values.size() == points.size(),
          ExcDimensionMismatch (values.size(), points.size()));
  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  const unsigned int n_points = points.size();
  for (unsigned int i=0; i<n_points; ++i)
    values[i] = value(points[i],component);
}


template <int dim>
Tensor<1,dim> CoefficientFun<dim>::gradient (const Point<dim>   &p,
                                             const unsigned int) const
{
  return Coefficient<dim>::gradient(p);
}


// function computing the right-hand side
template <int dim>
class RightHandSide : public Function<dim>
{
private:
  Solution<dim> solution;
  CoefficientFun<dim> coefficient;
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};


template <int dim>
double RightHandSide<dim>::value (const Point<dim>   &p,
                                  const unsigned int) const
{
  return -(solution.laplacian(p)*coefficient.value(p)
           + coefficient.gradient(p)*solution.gradient(p));
}

#endif /* _POISSON_COMMON_H */
