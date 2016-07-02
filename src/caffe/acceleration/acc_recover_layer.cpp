#include <algorithm>
#include <vector>

#include "caffe/acceleration/acc_recover_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"

namespace caffe {

template <typename Dtype>
void AccRecoverLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom[0]->num(), 1);
}

template <typename Dtype>
void AccRecoverLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int channels = bottom[0]->channels();
  top[0]->Reshape(bottom[0]->num(), channels == 1 ? 1 : channels-1, 
    bottom[0]->height(), bottom[0]->width());
  recovery_from_sparse(bottom[0]);
  caffe_copy(top[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void AccRecoverLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(AccRecoverLayer);
REGISTER_LAYER_CLASS(AccRecover);

}  // namespace caffe
