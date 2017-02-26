/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 *
 */

#ifndef _CUDA_MEMORY_UTILS_H
#define _CUDA_MEMORY_UTILS_H


namespace dealii
{
  namespace internal
  {

    template <typename DstNumber, typename SrcNumber>
    void copy_dev_array(DstNumber *dst,
                        const SrcNumber *src,
                        const unsigned len);
  }
}

#endif /* _CUDA_MEMORY_UTILS_H */
