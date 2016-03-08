/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 * @(#)timing.cu
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

double timer()
{
  cudaDeviceSynchronize();
  struct timespec ts;
  clock_gettime( CLOCK_MONOTONIC_RAW, &ts );
  return (double)ts.tv_sec +
    (double)ts.tv_nsec / 1000000000.0;
}
