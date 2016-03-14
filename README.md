This is a snapshot of the CUDA implementations of the matrix-free functionality of `Deal.II`, as of 2016-03-14.

It includes the general framework in the directory `matrix_free_gpu`, and an application `poisson.cu` which demonstrates the use of the framework. The application solves the Poisson equation with variable coefficient on a unit cube/square, using CG with a Chebyshev preconditioner. There is support for general grids with non-uniformly refined grids with hanging nodes although this is commented out in this version. `bmop.cu` deals with the exact same problem, but instead of solving it, it benchmarks the operator application by performing 100 successive applications. The subdirectory `opbm-cpu` implements the corresponding benchmark on CPU.

To build the application (require CUDA 7.5 and a CUDA-capable GPU):

   mkdir build
   cd build
   cmake -D CMAKE_CXX_FLAGS="--std=c++11 -march=native -DDEGREE_FE=4 -DDIMENSION=3 -DMATRIX_FREE_UNIFORM_MESH" -DCMAKE_BUILD_TYPE=RELEASE -DDEAL_II_DIR=/path/to/dealii ..
   make poisson
   ./poisson
