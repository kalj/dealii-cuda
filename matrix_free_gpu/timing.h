/* -*- c-basic-offset:4; tab-width:4; indent-tabs-mode:nil -*-
 *
 * @(#)timing.h
 * @author Karl Ljungkvist <karl.ljungkvist@it.uu.se>
 *
 */

#ifndef _TIMING_H
#define _TIMING_H

#include <time.h>

double timer();

#define START_TIMING(NREPS)                     \
    {                                           \
        double mintime = 9.0e100;               \
        for(int j = 0; j < NREPS; ++j) {        \
            double time=timer();


#define DONE_TIMING                                     \
            time=timer()-time;                          \
                                                        \
            mintime = time < mintime ? time : mintime;  \
        }                                               \
        printf("Elapsed time: %g s\n",mintime);          \
    }




#endif /* _TIMING_H */
