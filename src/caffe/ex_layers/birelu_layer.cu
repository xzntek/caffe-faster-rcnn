#include <algorithm>
#include <vector>

#include "caffe/ex_layers/birelu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BiReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype positive_slope, Dtype positive_thresh) {
  CUDA_KERNEL_LOOP(index, n) {
    if (in[index] <= Dtype(0)) {
      out[index] = negative_slope * in[index];
    } else if(in[index] <= positive_thresh) {
      out[index] = in[index];
    } else {
      out[index] = positive_thresh + (in[index]-positive_thresh) * positive_slope;
    }  
  }
}

template <typename Dtype>
void BiReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype negative_slope = this->layer_param_.bi_relu_param().negative_slope();
  const Dtype positive_slope = this->layer_param_.bi_relu_param().positive_slope();
  const Dtype positive_thresh = this->layer_param_.bi_relu_param().positive_thresh();
  // NOLINT_NEXT_LINE(whitespace/operators)
  BiReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, positive_slope, positive_thresh);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void BiReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype positive_slope, Dtype positive_thresh) {
  CUDA_KERNEL_LOOP(index, n) {
    //out_diff[index] = in_diff[index] * ((in_data[index] > 0)
    //    + (in_data[index] <= 0) * negative_slope);
    Dtype scale = 0;
    if (in_data[index] <= Dtype(0)) {
      scale = negative_slope;
    } else if(in_data[index] <= positive_thresh) {
      scale = 1;
    } else {
      scale = positive_slope;
    }
    out_diff[index] = in_diff[index] * scale;
  }
}

template <typename Dtype>
void BiReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype negative_slope = this->layer_param_.bi_relu_param().negative_slope();
    const Dtype positive_slope = this->layer_param_.bi_relu_param().positive_slope();
    const Dtype positive_thresh = this->layer_param_.bi_relu_param().positive_thresh();

    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    BiReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, positive_slope, positive_thresh);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(BiReLULayer);


}  // namespace caffe
