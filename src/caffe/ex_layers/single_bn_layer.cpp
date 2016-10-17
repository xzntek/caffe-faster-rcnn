#include <algorithm>
#include <vector>

#include "caffe/ex_layers/single_bn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SingleBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  if (param.has_use_global_stats()) CHECK(param.use_global_stats());
  CHECK_EQ(this->phase_, TEST);
  warm_up = 5;
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
  spatial_dim_ = bottom[0]->height() * bottom[0]->width();
  eps_ = param.eps();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0]=1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void SingleBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  mean_.Reshape(vector<int>(1, channels_));
  variance_.Reshape(vector<int>(1, channels_));
  
  bias_multiplier_.Reshape(vector<int>(1, spatial_dim_));
  if (bias_multiplier_.cpu_data()[spatial_dim_ - 1] != Dtype(1)) {
    caffe_set(spatial_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  CHECK_EQ(bottom[0]->count()/(channels_*bottom[0]->shape(0)), this->spatial_dim_);
  if (warm_up > 5) warm_up = 5;
}

template <typename Dtype>
void SingleBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(bottom[0]->shape(0), 1);

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  // use the stored mean/variance estimates.
  if (warm_up-->=0) {
  const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
          0 : 1 / this->blobs_[2]->cpu_data()[0];
  caffe_cpu_scale(variance_.count(), scale_factor,
          this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
  caffe_cpu_scale(variance_.count(), scale_factor,
          this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());
  }

  // subtract mean
  //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
  //    batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
  //    num_by_chans_.mutable_cpu_data());
  //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
  //    spatial_dim, 1, -1, num_by_chans_.cpu_data(),
  //    spatial_sum_multiplier_.cpu_data(), 1., top_data);
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, channels_,
    spatial_dim_, 1, Dtype(-1), mean_.cpu_data(),
    bias_multiplier_.cpu_data(), Dtype(1), top_data);

  for (int d = 0; d < channels_; d++) {
    const Dtype factor = 1 / variance_.cpu_data()[d];
    caffe_cpu_scale(spatial_dim_, factor, bottom_data, top_data);
    bottom_data += spatial_dim_;
    top_data += spatial_dim_;
  }
  // replicate variance to input size
}

INSTANTIATE_CLASS(SingleBatchNormLayer);
REGISTER_LAYER_CLASS(SingleBatchNorm);

}  // namespace caffe
