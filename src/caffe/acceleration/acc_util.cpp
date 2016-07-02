#include <vector>

#include "caffe/acceleration/acc_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void recovery_from_sparse(Blob<Dtype>* sparse) {
  CHECK_EQ(sparse->num(), 1);
  if (sparse->channels() == 1) return;
  const int channels = sparse->channels();
  const int spatial_dims = sparse->height() * sparse->width();
  const Dtype* Prod_ = sparse->cpu_data() + sparse->offset(0, channels-1, 0, 0);
  const int zeros = count_zeros(spatial_dims, Prod_);
  const int no_zeros = spatial_dims - zeros;
  if (zeros == 0) return;

  const Dtype* base_data = sparse->cpu_data() + (channels-1) * no_zeros;
  Dtype* ori_base = sparse->mutable_cpu_data() + (channels-1) * spatial_dims;

  for (int i = channels-1; i; i--) {
    const Dtype* prod_ = sparse->cpu_data() + sparse->offset(1);
    for (int count = spatial_dims; count; count--) {
      Dtype value = *(--prod_);
      if ( value == Dtype(0) ) {
        *(--ori_base) = 0;
      } else {
        *(--ori_base) = *(--base_data) * value;
      }
    }
  }
  caffe_set(spatial_dims, Dtype(1), sparse->mutable_cpu_data() + (channels-1) * spatial_dims);
}

template void recovery_from_sparse<float>(Blob<float>* sparse);
template void recovery_from_sparse<double>(Blob<double>* sparse);

}  // namespace caffe
