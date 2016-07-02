#include <vector>

#include "caffe/filler.hpp"
#include "caffe/acceleration/acc_bias_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccBiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // bias is a learned parameter; initialize it
    const BiasParameter& param = this->layer_param_.bias_param();
    CHECK_EQ(param.axis(), 1);
    CHECK_EQ(param.num_axes(), 1);
    const int axis = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis + num_axes)
          << "bias blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis;
    }
    this->blobs_.resize(1);

    vector<int> shape = bottom[0]->shape();
    CHECK_EQ(shape[1], bottom[0]->channels());
    shape[1] --;

    const vector<int>::const_iterator& shape_start =
        shape.begin() + axis;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? shape.end() : (shape_start + num_axes);
    vector<int> bias_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(param.filler()));
    filler->Fill(this->blobs_[0].get());
  } else {
    NOT_IMPLEMENTED;
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void AccBiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const BiasParameter& param = this->layer_param_.bias_param();
  Blob<Dtype>* bias = this->blobs_[0].get();
  // Always set axis == 0 in special case where bias is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis == 0 and (therefore) outer_dim_ == 1.
  const int axis = (bias->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis + bias->num_axes())
      << "bias blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis;

  vector<int> shape = bottom[0]->shape();
  CHECK_EQ(shape[1], bottom[0]->channels());
  shape[1] --;

  for (int i = 0; i < bias->num_axes(); ++i) {
    CHECK_LT(axis+i, shape.size());
    CHECK_EQ(shape[axis + i], bias->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis + i
        << ") and bias->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis);
  bias_dim_ = bias->count();
  inner_dim_ = bottom[0]->count(axis + bias->num_axes());
  //dim_ = bias_dim_ * inner_dim_;
  if (bottom[0] != top[0]) {
    top[0]->ReshapeLike(*bottom[0]);
  }
  bias_multiplier_.Reshape(vector<int>(1, inner_dim_));
  if (bias_multiplier_.cpu_data()[inner_dim_ - 1] != Dtype(1)) {
    caffe_set(inner_dim_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void AccBiasLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(outer_dim_, 1);
  const int spatial_dims = bottom[0]->height() * bottom[1]->width();
  CHECK_EQ(inner_dim_, spatial_dims);
  CHECK(bias_dim_ == 1 || bias_dim_ == bottom[0]->channels()-1);

  if (bottom[0]->channels() != 1) {
    const int zeros = count_zeros(spatial_dims, bottom[0]->cpu_data() +
        spatial_dims * bias_dim_);
    inner_dim_ = spatial_dims - zeros;
  }

  const Dtype* bias_data = this->blobs_[0].get()->cpu_data();
  
  Dtype* top_data = top[0]->mutable_cpu_data();
  if (bottom[0] != top[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, bias_dim_,
        inner_dim_, 1, Dtype(1), bias_data,
        bias_multiplier_.cpu_data(), Dtype(1), top_data);
}

template <typename Dtype>
void AccBiasLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(AccBiasLayer);
REGISTER_LAYER_CLASS(AccBias);

}  // namespace caffe
