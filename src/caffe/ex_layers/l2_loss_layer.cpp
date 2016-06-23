#include <vector>

#include "caffe/ex_layers/l2_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2LossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  //CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
  //    << "Inputs must have the same dimension.";
}

template <typename Dtype>
void L2LossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  this->dot = caffe_cpu_dot(count, bottom[0]->cpu_data(), bottom[0]->cpu_data());
  Dtype loss = sqrt(dot) / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void L2LossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //const Dtype sign = (i == 0) ? 1 : -1;
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num() / 2 / this->dot;
    caffe_cpu_axpby(
        bottom[0]->count(),              // count
        alpha,                              // alpha
        bottom[0]->cpu_data(),                   // a
        Dtype(0),                           // beta
        bottom[0]->mutable_cpu_diff());  // b
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2LossLayer);
#endif

INSTANTIATE_CLASS(L2LossLayer);
REGISTER_LAYER_CLASS(L2Loss);

}  // namespace caffe
