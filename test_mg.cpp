

#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/gpu_vec.h"


int main(int argc, char **argv)
{
  GpuVector<double> v(100);

  dealii::MGTransferMatrixFreeGpu<2,double> mg;

  mg.clear();

  return 0;
}
