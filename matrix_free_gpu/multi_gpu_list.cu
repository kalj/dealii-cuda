#include <deal.II/base/exceptions.h>

#include "multi_gpu_list.h"
#include "cuda_utils.cuh"
#include "cuda_memory_utils.h"

namespace dealii
{

  template <typename T>
  MultiGpuList<T>::MultiGpuList()
    : global_size(0),
      partitioner(new GpuPartitioner)
  {}

  template <typename T>
  MultiGpuList<T>::MultiGpuList(const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                                const std::vector<unsigned int> sizes)
    : partitioner(partitioner_in), local_sizes(sizes), values(sizes.size())
  {
    AssertDimension(partitioner->n_partitions(),sizes.size());

    global_size = 0;
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      global_size += local_sizes[i];

      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&values[i],local_sizes[i]*sizeof(T)));
    }
  }


  template <typename T>
  MultiGpuList<T>::MultiGpuList(const MultiGpuList<T> &other)
    : partitioner(other.partitioner), local_sizes(other.local_sizes),
      global_size(other.global_size), values(other.partitioner->n_partitions())
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&values[i],local_sizes[i]*sizeof(T)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(values[i],other.values[i],local_sizes[i]*sizeof(T),
                                    cudaMemcpyDeviceToDevice));
    }
  }

  template <typename T>
  template <typename T2>
  MultiGpuList<T>::MultiGpuList(const MultiGpuList<T2> &other)
    : partitioner(other.partitioner), local_sizes(other.local_sizes),
      global_size(other.global_size), values(other.partitioner->n_partitions())
  {
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&values[i],local_sizes[i]*sizeof(T)));

      internal::copy_dev_array(values[i],other.values[i],local_sizes[i]);
    }
  }


  template <typename T>
  void MultiGpuList<T>::reinit(const MultiGpuList<T> &other)
  {
    reinit(other.partitioner,other.local_sizes);
  }

  template <typename T>
  void MultiGpuList<T>::reinit(const std::shared_ptr<const GpuPartitioner> &partitioner_in,
                               const std::vector<unsigned int> sizes)
  {
    if(partitioner != NULL) {
      // first check if the partitioning needs to be adjusted
      if( partitioner_in->n_partitions() > partitioner->n_partitions()) {
        local_sizes.resize(partitioner_in->n_partitions(),0);
        values.resize(partitioner_in->n_partitions(), NULL);
      }
      else if(partitioner_in->n_partitions() < partitioner->n_partitions()) {
        local_sizes.resize(partitioner_in->n_partitions());
        // free up items to remove
        for(int i=partitioner_in->n_partitions(); i<partitioner->n_partitions(); ++i) {
          if(values[i] != NULL) {
            CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
            CUDA_CHECK_SUCCESS(cudaFree(values[i]));
            values[i] = NULL;
          }
        }
        values.resize(partitioner_in->n_partitions());
      }
    }
    else {
      local_sizes.resize(partitioner_in->n_partitions(),0);
      values.resize(partitioner_in->n_partitions(),NULL);
    }


    // now the number of partitions are right, but sizes are possibly wrong
    for(int i=0; i<partitioner_in->n_partitions(); ++i) {

      if(local_sizes[i] != sizes[i]) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        if(values[i] != NULL) {
          CUDA_CHECK_SUCCESS(cudaFree(values[i]));
        }
        CUDA_CHECK_SUCCESS(cudaMalloc(&values[i],sizes[i]*sizeof(T)));
        local_sizes[i] = sizes[i];
      }
    }

    // now assign partitioner and update global size
    partitioner = partitioner_in;
    global_size = 0;
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      global_size += local_sizes[i];
    }
  }

  template <typename T>
  MultiGpuList<T>::~MultiGpuList()
  {
    clear();
  }

  template <typename T>
  void MultiGpuList<T>::clear()
  {
    for(int i=0; i<values.size(); ++i) {
      if(values[i] != NULL) {
        CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
        CUDA_CHECK_SUCCESS(cudaFree(values[i]));
        values[i] = NULL;
      }
      local_sizes[i] = 0;
    }
    global_size = 0;
    partitioner = NULL;
  }

  template <typename T>
  void MultiGpuList<T>::swap(MultiGpuList<T> &other)
  {
    local_sizes.swap(other.local_sizes);
    values.swap(other.values);
    partitioner.swap(other.partitioner);

    std::swap(global_size,other.global_size);
  }

  template <typename T>
  template <typename T2>
  MultiGpuList<T>& MultiGpuList<T>::operator=(const MultiGpuList<T2> &other)
  {
    // make sure sizes are correct
    reinit(other.partitioner,other.local_sizes);

    // copy data
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMalloc(&values[i],local_sizes[i]*sizeof(T)));
      internal::copy_dev_array(values[i], other.values[i], local_sizes[i]);
    }
    return *this;
  }


  template <typename T>
  MultiGpuList<T>& MultiGpuList<T>::operator=(const MultiGpuList<T> &other)
  {
    // make sure sizes are correct
    reinit(other);

    // copy data
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(values[i],other.values[i],local_sizes[i]*sizeof(T),
                                    cudaMemcpyDeviceToDevice));
    }

    return *this;
  }

  template <typename T>
  MultiGpuList<T>& MultiGpuList<T>::operator=(const std::vector<T> &host_arr)
  {
    Assert(partitioner != NULL,ExcMessage("partitioner is not initialized"));
    AssertDimension(host_arr.size(),global_size);

    unsigned int offset = 0;
    for(int i=0; i<partitioner->n_partitions(); ++i) {
      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(values[i],host_arr.data()+offset,local_sizes[i]*sizeof(T),
                                    cudaMemcpyHostToDevice));
      offset += local_sizes[i];
    }

    return *this;
  }

  template <typename T>
  MultiGpuList<T>& MultiGpuList<T>::operator=(const std::vector<std::vector<T> > &host_arr)
  {
    Assert(partitioner != NULL,ExcMessage("partitioner is not initialized"));
    AssertDimension(host_arr.size(),partitioner->n_partitions());

    for(int i=0; i<partitioner->n_partitions(); ++i) {
      AssertDimension(host_arr[i].size(),local_sizes[i]);

      CUDA_CHECK_SUCCESS(cudaSetDevice(partitioner->get_partition_id(i)));
      CUDA_CHECK_SUCCESS(cudaMemcpy(values[i],host_arr[i].data(),local_sizes[i]*sizeof(T),
                                    cudaMemcpyHostToDevice));
    }

    return *this;
  }

  template <typename T>
  unsigned int MultiGpuList<T>::local_size(unsigned int part) const
  {
    return local_sizes[part];
  }

  template <typename T>
  unsigned int MultiGpuList<T>::size() const
  {
    return global_size;
  }

  template <typename T>
  const T* MultiGpuList<T>::getDataRO(unsigned int part) const
  {
    return values[part];
  }

  template <typename T>
  T* MultiGpuList<T>::getData(unsigned int part)
  {
    return values[part];
  }

  template <typename T>
  std::size_t MultiGpuList<T>::memory_consumption() const
  {
    return sizeof(T)*global_size;
  }

  template class MultiGpuList<unsigned int>;
  template class MultiGpuList<int>;
  template class MultiGpuList<float>;
  template class MultiGpuList<double>;

  template MultiGpuList<float>::MultiGpuList(const MultiGpuList<double>&);
  template MultiGpuList<double>::MultiGpuList(const MultiGpuList<float>&);
  template MultiGpuList<float>& MultiGpuList<float>::operator=(const MultiGpuList<double>&);
  template MultiGpuList<double>& MultiGpuList<double>::operator=(const MultiGpuList<float>&);
}