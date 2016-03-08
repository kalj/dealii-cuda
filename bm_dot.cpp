#include "gpu_vec.h"
#include "timing.h"

#include <deal.II/lac/vector.h>

using namespace dealii;

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
