#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/timing.h"

#include <deal.II/lac/vector.h>

using namespace dealii;


/*
  Some benchmark results:

  Performance serial reduction:
  bksize        time
  1024          1.03
  512           0.426
  256           0.433
  128           0.79
  64            1.76
  32            3.4
  16            6.8

  Performance parallel reduction, every s'th thread
  1024          0.25
  512           0.202
  256           0.48

  Performance parallel reduction, the bksize/s first threads
  1024          0.203
  512           0.206
  256           0.499

  Performance -||-, improved looping
  1024          0.2003

  Performance chunking (cs=2)
  1024          0.131
  512           0.124
  256           0.272

  Performance chunking (cs=4)
  1024          0.103
  512           0.085
  256           0.153

  Performance chunking (cs=8)
  1024          0.100
  512           0.083
  256           0.093

  Performance chunking (cs=16)
  1024          0.091
  512           0.084
  256           0.084

  Performance (cs=8), unroll from 32
  1024          0.097
  512           0.083
  256           0.092

*/

int main(int argc, char *argv[])
{
    typedef double number;

    const int N=1024000;
    const int Nreps=100;


    Vector<number> v1(N);
    Vector<number> v2(N);

    for(int i = 0; i < N; ++i) {
        v1(i) = 1.0/N;
        v2(i) = 1.0/(i+1);
    }


    GpuVector<number> gv1(v1);
    GpuVector<number> gv2(v2);

    printf("Verification:\n");
    printf("  cpu:  %10g\n",(v1*v2));
    printf("  gpu:  %10g\n",(gv1*gv2));
    printf("  diff: %10g\n",(v1*v2)-(gv1*gv2));


    number d = 1.0;
    double t = timer();
    for(int i = 0; i < Nreps; ++i) {


        d *= gv1*gv2;
        // d *= dot(v1,v2);
    }
    printf("Elapsed time: %g s\n",timer()-t);

    return int(d);
}
