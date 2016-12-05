
#ifndef _POISSON_COMMON_H
#define _POISSON_COMMON_H

#include <deal.II/base/vectorization.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include "matrix_free_gpu/maybecuda.h"
#include "matrix_free_gpu/gpu_array.cuh"

using namespace dealii;
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


//=============================================================================
// coefficient
//=============================================================================

template <int dim>
struct Coefficient
{
  static MAYBECUDA_HOST double value (const Point<dim> &p){
    return 1. / (0.05 + 2.*(p.norm_square()));
    // return 1.;
  }

#ifdef MAYBECUDA_GUARD
  template <typename Number>
  static __device__ Number value (const GpuArray<dim,Number> &p){
    return 1. / (0.05 + 2.*(p.norm_square()));
    // return 1.;
  }
#endif

  template <typename Number>
  static VectorizedArray<Number> value (const Point<dim,VectorizedArray<Number>> &p){
    return 1. / (0.05 + 2.*(p.norm_square()));
    // VectorizedArray<Number> a;
    // a = 1.0;
    // return a;
  }


  static MAYBECUDA_HOST  Tensor<1,dim> gradient (const Point<dim> &p){
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
  return Coefficient<dim>::value(p);
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
