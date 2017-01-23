#include "gpu_list.h"

#include "cuda_utils.cuh"

template <typename T>
GpuList<T>::GpuList()
{
  n = 0;
  values = NULL;
}

template <typename T>
GpuList<T>::GpuList(const GpuList<T> &other)
{
  n = other.size();
  cudaMalloc(&values,n*sizeof(T));
  cudaAssertNoError();
  cudaMemcpy(values,other.values,n*sizeof(T),
             cudaMemcpyDeviceToDevice);
  cudaAssertNoError();
}

template <typename T>
GpuList<T>::GpuList(const std::vector<T> &host_arr)
{
  n = host_arr.size();
  cudaMalloc(&values,n*sizeof(T));
  cudaAssertNoError();
  cudaMemcpy(values,&host_arr[0],n*sizeof(T),
             cudaMemcpyHostToDevice);
  cudaAssertNoError();
}

template <typename T>
GpuList<T>::~GpuList()
{
  if(values != NULL) {
    cudaFree(values);
    cudaAssertNoError();
  }
}

template <typename T>
void GpuList<T>::resize(unsigned int newsize)
{
  if(n != newsize)
  {
    if(values != NULL) {
      cudaFree(values);
      cudaAssertNoError();
    }

    n = newsize;
    cudaMalloc(&values,n*sizeof(T));
    cudaAssertNoError();
  }
}

template <typename T>
GpuList<T>& GpuList<T>::operator=(const GpuList<T> &other)
{
  resize(other.size());

  cudaMemcpy(values,other.values,n*sizeof(T),
             cudaMemcpyDeviceToDevice);
  cudaAssertNoError();

  return *this;
}

  template <typename T>
  GpuList<T>& GpuList<T>::operator=(const std::vector<T> &host_arr)
  {

    resize(host_arr.size());

    cudaMemcpy(values,host_arr.data(),n*sizeof(T),
               cudaMemcpyHostToDevice);
    cudaAssertNoError();

    return *this;
  }

    template <typename T>
    void GpuList<T>::clear()
    {
      n = 0;
      if(values != NULL) {
        cudaFree(values);
        cudaAssertNoError();
        values = NULL;
      }
    }

template <typename T>
unsigned int GpuList<T>::size() const
{
  return n;
}

template <typename T>
const T* GpuList<T>::getDataRO() const
{
  return values;
}

template <typename T>
std::size_t GpuList<T>::memory_consumption() const
{
  return sizeof(T)*n;
}

template class GpuList<unsigned int>;
template class GpuList<int>;
template class GpuList<float>;
template class GpuList<double>;