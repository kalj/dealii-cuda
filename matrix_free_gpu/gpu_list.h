#ifndef _GPU_LIST_H
#define _GPU_LIST_H

#include <vector>

template <typename T>
class GpuList {
private:
  unsigned int n;
  T *values;
public:
  GpuList();

  GpuList(const GpuList<T> &other);

  GpuList(const std::vector<T> &host_arr);

  ~GpuList();

  void resize(unsigned int newsize);

  GpuList<T> & operator=(const GpuList<T> &other);

  GpuList<T> & operator=(const std::vector<T> &host_arr);

  void clear();

  unsigned int size() const;

  const T* getDataRO() const;

  std::size_t memory_consumption() const;

};

#endif /* _GPU_LIST_H */
