#include <algorithm>
#include <vector>

#include "caffe/acceleration/acc_relu_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"

namespace caffe {

template <typename Dtype>
void AccReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  CHECK_EQ(negative_slope, Dtype(0));
  CHECK_EQ(bottom[0]->num(), 1);
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
}

template <typename Dtype>
void AccReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Reshape Done in Neuron Layer
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const int count = bottom[0]->count();
  if (bottom[0]->channels() == 1) {
    for (int i = 0; i < count; i++) {
      top_data[i] = std::max(Dtype(0), bottom_data[i]);
    }
  } else {
    const int prod_count = bottom[0]->height() * bottom[0]->width();
    const Dtype* Prod_ = bottom[0]->cpu_data() + bottom[0]->offset(0, bottom[0]->channels()-1, 0, 0);
    const int zeros = count_zeros(prod_count, Prod_);

    CHECK_GE(prod_count, zeros);
    for (int i = 0; i < (prod_count-zeros)*(bottom[0]->channels()-1); i++) {
      top_data[i] = std::max(Dtype(0), bottom_data[i]);
    }
    for (int i = count - prod_count; i < count; i++) {
      top_data[i] = bottom_data[i];
    }
  }
}

template <typename Dtype>
void AccReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}


INSTANTIATE_CLASS(AccReLULayer);
REGISTER_LAYER_CLASS(AccReLU);

}  // namespace caffe
