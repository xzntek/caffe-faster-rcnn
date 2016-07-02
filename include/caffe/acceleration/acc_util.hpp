#ifndef _CAFFE_UTIL_ACC_HPP_
#define _CAFFE_UTIL_ACC_HPP_

#include "caffe/blob.hpp"

namespace caffe {

template <typename Dtype>
inline int count_zeros(int count, const Dtype* data) {
  int zeros = 0;
  while (count--) {
    zeros += *(data++) == Dtype(0);
  }
  return zeros;
}

template <typename Dtype>
void recovery_from_sparse(Blob<Dtype>* sparse);

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
