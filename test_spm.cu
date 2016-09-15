
#include <deal.II/lac/vector.h>
#include "matrix_free_gpu/cuda_sparse_matrix.h"

using namespace dealii;

int main(int argc, char **argv)
{

  ::CUDAWrappers::SparseMatrix<double> m;

  m.init();

  m.print();

  Vector<double> v(4);

  v[0] = 1.3;
  v[1] = 0.3;
  v[2] = 1.9;
  v[3] = -6.3;



  GpuVector<double> gv(v);

  GpuVector<double> gv2(4);


  m.vmult(gv2,gv);

  gv2.copyToHost(v);


  printf("result:\n");
  v.print(std::cout, 3, false, false);



  return 0;
}