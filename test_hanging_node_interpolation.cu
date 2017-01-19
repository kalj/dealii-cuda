/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)test_hanging_node_interpolation.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#include <vector>
#include <iostream>
#include <bitset>
#include <boost/algorithm/string/join.hpp>

#include <deal.II/fe/fe_tools.h>

#include "matrix_free_gpu/hanging_nodes.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/utils.h"


using namespace dealii;

typedef double Number;


template <unsigned int dim, unsigned int fe_degree>
__global__ void shmem_kernel(Number *vec, const unsigned int mask)
{
  const unsigned int n_dofs_1d = fe_degree+1;
  const unsigned int n_dofs = ipowf(n_dofs_1d,dim);

  __shared__ Number shmem[n_dofs];
  const unsigned int idx = dim==3 ? threadIdx.x + threadIdx.y*n_dofs_1d + threadIdx.z*n_dofs_1d*n_dofs_1d :
    threadIdx.x + threadIdx.y*n_dofs_1d;

  shmem[idx] = vec[idx];
  __syncthreads();

  resolve_hanging_nodes_shmem<dim,fe_degree,NOTRANSPOSE> (shmem,mask);

  __syncthreads();
  vec[idx] = shmem[idx];


}


template <unsigned int dim, unsigned int fe_degree>
void print(Number *vec_host)
{
  const unsigned int n_dofs_1d = fe_degree+1;

  if(dim==2) {
    for(int i = 0; i < n_dofs_1d; ++i) {
      printf("[");
      for(int j = 0; j < n_dofs_1d; ++j)
        printf("\t%8.3g",vec_host[i*n_dofs_1d + j]);

      if(i != n_dofs_1d-1) printf("\n");
    }
    printf("]\n");
  }
  else if(dim==3) {
    for(int i = 0; i < n_dofs_1d; ++i) {
      printf("[");
      for(int j = 0; j < n_dofs_1d; ++j) {
        for(int k = 0; k < n_dofs_1d; ++k)
          printf("\t%8.3g",vec_host[i*n_dofs_1d*n_dofs_1d + j*n_dofs_1d + k]);

        if(j != n_dofs_1d-1) printf("\n");
      }
      printf("]\n");
    }
  }
}

Number foo(Number x, Number y, Number z) {
  // linear polynomial
  Number a = x+2*y+3*z;
  // cubic polynomial
  // Number a = (x*2-x*x*x)*(5*y*y*y+y-y*y-1)*(3*z*z+1);
  return a;
}


template <unsigned int dim, unsigned int fe_degree>
void setup_values(Number * vec, const unsigned int mask)
{
  const unsigned int n_dofs_1d = fe_degree+1;

  FE_Q<dim> fe_q(fe_degree);
  const std::vector< Point< dim > > &support_points = fe_q.get_unit_support_points ();
  std::vector<unsigned int> index_mapping(fe_q.dofs_per_cell);
  FETools::lexicographic_to_hierarchic_numbering<dim>(fe_q,index_mapping);

  if(dim==2) {
    if(mask & CONSTR_FACE_X) {

      for(int i = 0; i < n_dofs_1d; ++i) {
        unsigned int j= (mask & CONSTR_TYPE_X)? 0 : fe_degree;

        const int idx = j + i*n_dofs_1d;
        const Point<dim> &p = support_points[index_mapping[idx]];

        Number x = p(0);
        Number y = 2*p(1);

        if(!(mask & CONSTR_TYPE_Y))
          y -= 1.0;

        vec[idx] = foo(x,y,1);
      }
    }

    if(mask & CONSTR_FACE_Y) {

      for(int j = 0; j < n_dofs_1d; ++j) {
        unsigned int i= (mask & CONSTR_TYPE_Y)? 0 : fe_degree;

        const int idx = j + i*n_dofs_1d;
        const Point<dim> &p = support_points[index_mapping[idx]];

        Number x = 2*p(0);
        Number y = p(1);

        if(!(mask & CONSTR_TYPE_X))
          x -= 1.0;
        vec[idx] = foo(x,y,1);
      }
    }
  }
  else { // 3D
    // yz plane
    if(mask & CONSTR_FACE_X) {

      for(int i = 0; i < n_dofs_1d; ++i) {
        for(int j = 0; j < n_dofs_1d; ++j) {
          unsigned int k= (mask & CONSTR_TYPE_X)? 0 : fe_degree;

          const int idx = k+ n_dofs_1d*(j + i*n_dofs_1d);
          const Point<dim> &p = support_points[index_mapping[idx]];

          Number x = p(0);
          Number y = 2*p(1);
          Number z = 2*p(2);

          if(!(mask & CONSTR_TYPE_Y))
            y -= 1.0;

          if(!(mask & CONSTR_TYPE_Z))
            z -= 1.0;

          vec[idx] = foo(x,y,z);
        }
      }
    }

    // xz plane
    if(mask & CONSTR_FACE_Y) {

      for(int i = 0; i < n_dofs_1d; ++i) {
        for(int k = 0; k < n_dofs_1d; ++k) {
          unsigned int j= (mask & CONSTR_TYPE_Y)? 0 : fe_degree;

          const int idx = k+ n_dofs_1d*(j + i*n_dofs_1d);
          const Point<dim> &p = support_points[index_mapping[idx]];

          Number x = 2*p(0);
          Number y = p(1);
          Number z = 2*p(2);

          if(!(mask & CONSTR_TYPE_X))
            x -= 1.0;

          if(!(mask & CONSTR_TYPE_Z))
            z -= 1.0;

          vec[idx] = foo(x,y,z);
        }
      }
    }

    // xy plane
    if(mask & CONSTR_FACE_Z) {

      for(int j = 0; j < n_dofs_1d; ++j) {
        for(int k = 0; k < n_dofs_1d; ++k) {
          unsigned int i= (mask & CONSTR_TYPE_Z)? 0 : fe_degree;

          const int idx = k+ n_dofs_1d*(j + i*n_dofs_1d);
          const Point<dim> &p = support_points[index_mapping[idx]];

          Number x = 2*p(0);
          Number y = 2*p(1);
          Number z = p(2);

          if(!(mask & CONSTR_TYPE_X))
            x -= 1.0;

          if(!(mask & CONSTR_TYPE_Y))
            y -= 1.0;

          vec[idx] = foo(x,y,z);
        }
      }
    }
  }
}

template <unsigned int dim, unsigned int fe_degree>
double test_constraint(unsigned int mask,
                       bool verbose)
{
  const unsigned int n_dofs_1d = fe_degree+1;
  const unsigned int n_dofs = ipowf(n_dofs_1d,dim);

  setup_constraint_weights<dim>(fe_degree);

  Number *vec_dev;
  Number *vec_host = new Number[n_dofs];
  Number *ref = new Number[n_dofs];
  const unsigned int size = n_dofs*sizeof(Number);

  // intialize data
  FE_Q<dim> fe_q(fe_degree);
  const std::vector< Point< dim > > &support_points = fe_q.get_unit_support_points ();
  std::vector<unsigned int> index_mapping(fe_q.dofs_per_cell);
  FETools::lexicographic_to_hierarchic_numbering<dim>(fe_q,index_mapping);

  for(int i=0; i<fe_q.dofs_per_cell; i++) {
    const Point<dim> &p = support_points[index_mapping[i]];

    const double x = p(0);
    const double y = p(1);
    const double z = dim==3 ? p(2) : 1.0;
    vec_host[i] = foo(x,y,z);
  }

  memcpy(ref,vec_host,size);

  setup_values<dim,fe_degree>(vec_host,mask);

  if(verbose) {
    printf("Reference:\n");
    print<dim,fe_degree>(ref);

    printf("Input:\n");
    print<dim,fe_degree>(vec_host);

  }

  CUDA_CHECK_SUCCESS(cudaMalloc(&vec_dev,size));

  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_dev,vec_host,size,cudaMemcpyHostToDevice));

    dim3 grid_dim(1);
    dim3 block_dim;
    if(dim==3) block_dim = dim3(n_dofs_1d,n_dofs_1d,n_dofs_1d);
    else if(dim==2) block_dim = dim3(n_dofs_1d,n_dofs_1d);

    shmem_kernel<dim,fe_degree> <<<grid_dim, block_dim >>> (vec_dev,mask);


  CUDA_CHECK_SUCCESS(cudaMemcpy(vec_host,vec_dev,size,cudaMemcpyDeviceToHost));

  // print data

  double err = 0;
  for(int i = 0; i < n_dofs; ++i) {
    double d = ref[i]-vec_host[i];
    ref[i] = d*d;
    err += d*d;
  }

  if(verbose) {
    printf("Result:\n");
    print<dim,fe_degree>(vec_host);

    printf("Difference:\n");
    print<dim,fe_degree>(ref);
  }

  CUDA_CHECK_SUCCESS(cudaFree(vec_dev));
  delete[] vec_host;
  delete[] ref;

  return err;
}

void print_mask(unsigned int mask)
{

  std::vector<std::string> strvec;

  if(mask & CONSTR_FACE_X)
    strvec.push_back("CONSTR_FACE_X");
  if(mask & CONSTR_FACE_Y)
    strvec.push_back("CONSTR_FACE_Y");
  if(mask & CONSTR_FACE_Z)
    strvec.push_back("CONSTR_FACE_Z");
  if(mask & CONSTR_TYPE_X)
    strvec.push_back("CONSTR_TYPE_X");
  if(mask & CONSTR_TYPE_Y)
    strvec.push_back("CONSTR_TYPE_Y");
  if(mask & CONSTR_TYPE_Z)
    strvec.push_back("CONSTR_TYPE_Z");

  std::string joined = boost::algorithm::join(strvec," | ");
  std::cout << joined << std::endl;
}

template <int dim, int p>
void loop_over_constraints(bool combinations)
{
  printf("---- %dD, order %d ----\n",dim,p);

  int n = 0;
  bool allpass=true;
  double toterr = 0;
  int xyz_max = combinations ? ((1<<dim)-1) : dim ;

  for(int xyz = 0; xyz < xyz_max; ++xyz) {

    for(int i = 0; i < (1<<dim); ++i) {
      n++;
      unsigned int mask = i; // set the type bits
      // now set the face
      if(combinations)
        mask |= (xyz<<3);
      else
        mask |= 1<<(xyz+3);

      double err = test_constraint<dim,p>(mask,
                                          false);

      // printf(" Constraint: ");
      // print_mask(mask);
      // std::cout << "mask: " << std::bitset<6>(mask) << std::endl;
      // printf("  Error: %g\n",err);
      // printf("\n ");

      if(err > 0) {
        toterr+=err;
        allpass = false;
      }
    }
  }
  if(allpass) printf(" (all %d tests gave 0 error)\n",n);
  else printf(" Total error from %d tests: %g\n",n,toterr);
  printf("\n");
}

int main(int argc, char *argv[])
{
  bool combinations = true;

  printf("======================================================================\n");
  printf("= Performing tests for version using parallelization within elements =\n");
  printf("======================================================================\n");

  loop_over_constraints<2,1>(/*pmem,*/combinations);
  loop_over_constraints<2,2>(/*pmem,*/combinations);
  loop_over_constraints<2,3>(/*pmem,*/combinations);
  loop_over_constraints<2,4>(/*pmem,*/combinations);

  loop_over_constraints<3,1>(/*pmem,*/combinations);
  loop_over_constraints<3,2>(/*pmem,*/combinations);
  loop_over_constraints<3,3>(/*pmem,*/combinations);
  loop_over_constraints<3,4>(/*pmem,*/combinations);

  return 0;
}
