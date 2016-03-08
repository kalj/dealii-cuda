#include "gpu_array.cuh"
#include "cuda_utils.cuh"
#include <cassert>

template <int dim, typename Number>
struct Coefficient
{
    __device__ static Number value (const GpuArray<dim,Number> &p){

        return 1. / (0.05 + 2.*(p.norm_square()));
    }

    __device__ static void value_list (const GpuArray<dim,Number> *points,
                                       Number                     *values,
                                       const unsigned int         len) {
    for (unsigned int i=0; i<len; ++i)
        values[i] = value(points[i]);
    }
};

template <int dim, typename Number, int npts, typename coefficient_function>
__global__ void coeff_eval (Number                          *coeff,
                            const GpuArray<dim,Number> *points,
                            const int ncells)
{
    const unsigned int cell = threadIdx.x + blockDim.x*blockIdx.x;

    if(cell < ncells) {

        for (unsigned int q=0; q<npts; ++q)
            coeff[cell*npts+q] = coefficient_function::value(points[cell*npts+q]);
    }
}



int main ()
{
    const int dim = 2;
    typedef double Number;
    typedef GpuArray<dim,Number> point_type;

    const int N_pts = 8;
    const int N_cells = 8;

    point_type *quad_points_host = new point_type[N_cells*N_pts];

    for(int i=0; i< N_cells; ++i) {
        for(int q=0; q< N_pts; ++q) {
            quad_points_host[i*N_pts+q].arr[0] = i*100+q;
            quad_points_host[i*N_pts+q].arr[1] = i*100+q+0.3;
        }
    }

    point_type *quad_points_dev;

    CUDA_CHECK_SUCCESS(cudaMalloc(&quad_points_dev, N_pts*N_cells*sizeof(point_type)));
    CUDA_CHECK_SUCCESS(cudaMemcpy(quad_points_dev, quad_points_host, N_pts*N_cells*sizeof(point_type),
                                  cudaMemcpyHostToDevice));

    Number *value_dev;
    Number *value_host = new Number[N_pts*N_cells];

    CUDA_CHECK_SUCCESS(cudaMalloc(&value_dev, N_pts*N_cells*sizeof(Number)));



    coeff_eval<dim,Number,N_pts,Coefficient<dim,Number> > <<<1,N_cells>>> (value_dev,quad_points_dev,N_cells);




    CUDA_CHECK_SUCCESS(cudaMemcpy(value_host, value_dev, N_pts*N_cells*sizeof(Number),
                                  cudaMemcpyDeviceToHost));

    for(int i=0; i< N_pts; ++i) {
        printf("val[%d]: ",i);
        for (unsigned int q=0; q<N_pts; ++q)
            printf("%12.7g",value_host[i*N_pts+q]);
        printf("\n");
    }

    CUDA_CHECK_SUCCESS(cudaFree(quad_points_dev));
    CUDA_CHECK_SUCCESS(cudaFree(value_dev));

    delete[] quad_points_host;
    delete[] value_host;
    return 0;
}