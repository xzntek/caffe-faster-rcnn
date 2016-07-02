#include <algorithm>
#include <vector>

#include "caffe/acceleration/acc_batch_norm_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  CHECK_EQ(param.has_use_global_stats(), false);

  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1) - 1;

  if (channels_ == 0)
    channels_ = 1; // channels == 1

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
void AccBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), 1);
  vector<int> shape = bottom[0]->shape();
  if (shape.size() > 1 && shape[1] > 1)
    shape[1] --;

  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(shape[1], channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  //x_norm_.ReshapeLike(*bottom[0]);
  sz[0]=bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->height()*bottom[0]->width();
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void AccBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  CHECK_EQ(bottom[0]->shape(0), 1);
  int spatial_dim = bottom[0]->height()*bottom[0]->width();

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }

  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
    caffe_cpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
  } else {
    NOT_IMPLEMENTED;
  }

  // subtract mean
//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
//      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
//      num_by_chans_.mutable_cpu_data());
  CHECK_EQ(num_by_chans_.count(), mean_.count());
  if (bottom[0]->channels() > 1) {
    const int zeros = count_zeros(spatial_dim, bottom[0]->cpu_data() + 
        bottom[0]->offset(0, channels_));
    spatial_dim -= zeros;
  }

//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
//      spatial_dim, 1, -1, num_by_chans_.cpu_data(),
//      spatial_sum_multiplier_.cpu_data(), 1., top_data);
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
      spatial_dim, 1, -1, mean_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., top_data);

  // normalize variance
  caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
  caffe_powx(variance_.count(), variance_.cpu_data(), Dtype(0.5),
             variance_.mutable_cpu_data());

  // replicate variance to input size
//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
//      batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
//      num_by_chans_.mutable_cpu_data());
//  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
//      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
//      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_,
      spatial_dim, 1, 1., variance_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
  caffe_div(channels_*spatial_dim, top_data, temp_.cpu_data(), top_data);
  
  if (bottom[0]->channels() > 1) {
    spatial_dim = bottom[0]->height()*bottom[0]->width();
    caffe_copy(spatial_dim, bottom[0]->cpu_data() + channels_ * spatial_dim,
        top[0]->mutable_cpu_data() + channels_ * spatial_dim);
  }
}

template <typename Dtype>
void AccBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(AccBatchNormLayer);
REGISTER_LAYER_CLASS(AccBatchNorm);
}  // namespace caffe
