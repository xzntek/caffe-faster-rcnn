#include <vector>

#include "caffe/ex_layers/l2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(), &this->dot);
  Dtype loss = sqrt(dot) / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L2LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num() / 2 / this->dot;
    caffe_gpu_axpby(
        bottom[0]->count(),              // count
        alpha,                              // alpha
        bottom[0]->gpu_data(),               // a
        Dtype(0),                           // beta
        bottom[0]->mutable_gpu_diff());  // b
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2LossLayer);

}  // namespace caffe
