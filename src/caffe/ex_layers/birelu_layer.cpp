#include <algorithm>
#include <vector>

#include "caffe/ex_layers/birelu_layer.hpp"

namespace caffe {

template <typename Dtype>
void BiReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.bi_relu_param().negative_slope();
  const Dtype positive_slope = this->layer_param_.bi_relu_param().positive_slope();
  const Dtype positive_thresh = this->layer_param_.bi_relu_param().positive_thresh();
  CHECK_GT(positive_thresh, Dtype(0));

  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] <= Dtype(0)) {
      top_data[i] = negative_slope * bottom_data[i];
    } else if(bottom_data[i] <= positive_thresh) {
      top_data[i] = bottom_data[i];
    } else {
      top_data[i] = positive_thresh + (bottom_data[i]-positive_thresh) * positive_slope;
    }
  }

}

template <typename Dtype>
void BiReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype negative_slope = this->layer_param_.bi_relu_param().negative_slope();
    const Dtype positive_slope = this->layer_param_.bi_relu_param().positive_slope();
    const Dtype positive_thresh = this->layer_param_.bi_relu_param().positive_thresh();

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      Dtype scale = 0;
      if (bottom_data[i] <= Dtype(0)) {
        scale = negative_slope;
      } else if(bottom_data[i] <= positive_thresh) {
        scale = 1;
      } else {
        scale = positive_slope;
      }
      bottom_diff[i] = top_diff[i] * scale;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BiReLULayer);
#endif

INSTANTIATE_CLASS(BiReLULayer);
REGISTER_LAYER_CLASS(BiReLU);

}  // namespace caffe
