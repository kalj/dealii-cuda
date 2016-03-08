
#include <iostream>
#include "matrix_free_gpu/gpu_vec.h"
#include <deal.II/lac/vector.h>

using namespace dealii;

#define MYASSERT(expr,text) do {             \
        if(expr) printf("Test %s: PASS\n",text);        \
        else printf("Test %s: FAILED\n",text);          \
    } while(0);


int main(int argc, char *argv[])
{
    const int N=5;
    Vector<double> v1(N);
    Vector<double> v2(N);

    for(int i = 0; i < N; ++i) {
        v1(i) = i;
        v2(i) = 1+1.0/(i+1);
    }

    // std::cout << v1 << std::endl;
    // std::cout << v2 << std::endl;

    GpuVector<double> gv1(v1);
    MYASSERT(gv1.size() == N,"size==5");
    GpuVector<double> gv2(v2);
    MYASSERT(gv2.size() == N,"size==5");

    GpuVector<double> gv3(10);
    gv3.reinit(gv1);
    MYASSERT(gv3.size() == N,"\"reinit\"");

    printf("--- dot product ---\n");
    double cpudot = v1*v2;
    double gpudot = gv1*gv2;

    printf("diff: %10g\n",cpudot-gpudot);

    printf("--- norm ---\n");
    printf("cpunorm: %g\n",v1.l2_norm());
    printf("gpunorm: %g\n",gv1.l2_norm());


    printf("--- addition ---\n");
    gv1.sadd(3,4,gv2);
    gv1 *= 1.3;

    v1.sadd(3,4,v2);
    v1 *= 1.3;

    std::cout << "1.3*(v1*3 + v2*4):" << std::endl
              << v1 << std::endl;
    std::cout << "1.3*(gv1*3 + gv2*4):" << std::endl
              << gv1.toVector() << std::endl;
    Vector<double> gv_host = gv1.toVector();
    v1 -= gv_host;
    printf("diffnorm: %g\n",(v1).l2_norm());

    printf("--- add_and_dot ----\n");

    Vector<double> v3(N);

    for(int i = 0; i < N; ++i) {
        v1(i) = i;
        v2(i) = 1+1.0/(i+1);
        v3(i) = -1;
    }

    gv1 = v1;
    gv2 = v2;
    gv3 = v3;

    double res = v1.add_and_dot(1.4,v2,v3);

    printf("res: %g\n",res);
    std::cout << v1 << std::endl;

    double gres = gv1.add_and_dot(1.4,gv2,gv3);
    printf("gres: %g\n",gres);
    std::cout << gv1.toVector() << std::endl;



    printf("---- larger vectors... ----\n");


    v1.reinit(10000);
    v2.reinit(10000);
    v3.reinit(10000);

    srand48(0);

    for(int i = 0; i < 10000; ++i) {
      v1(i) = drand48();
      v2(i) = drand48();
      v3(i) = drand48();
    }

    gv1 = v1;
    gv2 = v2;
    gv3 = v3;

    res = v1.add_and_dot(1.4,v2,v3);
    printf("res: %g\n",res);

    gres = gv1.add_and_dot(1.4,gv2,gv3);
    printf("gres: %g\n",gres);


    return 0;
}
