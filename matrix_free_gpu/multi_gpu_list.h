#ifndef _MULTI_GPU_LIST_H
#define _MULTI_GPU_LIST_H

#include <vector>
#include <memory>
#include "gpu_partitioner.h"


namespace dealii
{
  template <typename T>
  class MultiGpuList {
  private:
    std::vector<unsigned int> local_sizes;
    unsigned int global_size;
    std::vector<T *> values;
    std::shared_ptr<const GpuPartitioner> partitioner;
  public:
    MultiGpuList();

    MultiGpuList(const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                 const std::vector<unsigned int> sizes);

    MultiGpuList(const MultiGpuList<T> &other);

    template <typename T2>
    MultiGpuList(const MultiGpuList<T2> &other);

    ~MultiGpuList();

    void reinit(const MultiGpuList<T> &other);

    void reinit(const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                const std::vector<unsigned int> sizes);

    MultiGpuList<T> & operator=(const MultiGpuList<T> &other);

    template <typename T2>
    MultiGpuList<T> & operator=(const MultiGpuList<T2> &other);

    MultiGpuList<T> & operator=(const std::vector<T> &host_arr);

    MultiGpuList<T> & operator=(const std::vector<std::vector<T> > &host_arr);

    // void set_local(const std::vector<T> &host_arr);

    void clear();

    unsigned int local_size(unsigned int part) const;

    unsigned int size() const;

    const T* getDataRO(unsigned int part) const;

    T* getData(unsigned int part);

    std::size_t memory_consumption() const;

    template <typename OtherNumber> friend class MultiGpuList;

  };
}

#endif /* _MULTI_GPU_LIST_H */
