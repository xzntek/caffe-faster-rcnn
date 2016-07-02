#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/acceleration/acc_scale_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  CHECK_EQ(param.axis(), 1);
  CHECK_EQ(param.num_axes(), 1);
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(1);

    vector<int> shape = bottom[0]->shape();
    CHECK_EQ(shape[1], bottom[0]->channels());
    shape[1] --;

    const vector<int>::const_iterator& shape_start =
        shape.begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  } else {
    NOT_IMPLEMENTED;
  }

  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("AccBias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    bias_param->set_num_axes(param.num_axes());

    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    bias_layer_->SetUp(bias_bottom_vec_, top);
    bias_param_id_ = this->blobs_.size();
    this->blobs_.resize(bias_param_id_ + 1);
    this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void AccScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;

  vector<int> shape = bottom[0]->shape();
  CHECK_EQ(shape[1], bottom[0]->channels());
  shape[1] --;
  
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_LT(axis_+i, shape.size());
    CHECK_EQ(shape[axis_ + i], scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());

  top[0]->ReshapeLike(*bottom[0]);
  if (bias_layer_) {
    bias_bottom_vec_[0] = top[0];
    bias_layer_->Reshape(bias_bottom_vec_, top);
  }
}

template <typename Dtype>
void AccScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int spatial_dims = bottom[0]->height() * bottom[1]->width();
  CHECK_EQ(inner_dim_, spatial_dims);
  CHECK_EQ(outer_dim_, 1);
  CHECK(scale_dim_==1 || scale_dim_==bottom[0]->channels()-1);

  if (bottom[0]->channels() != 1) {
    const int zeros = count_zeros(spatial_dims, bottom[0]->cpu_data() + 
        spatial_dims * scale_dim_);
    inner_dim_ = spatial_dims - zeros;
  }

  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* scale_data = this->blobs_[0].get()->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int d = 0; d < scale_dim_; ++d) {
    const Dtype factor = scale_data[d];
    caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
    bottom_data += inner_dim_;
    top_data += inner_dim_;
  }
  caffe_copy(spatial_dims, bottom_data + spatial_dims * scale_dim_, 
        top_data+ spatial_dims * scale_dim_);

  if (bias_layer_) {
    bias_layer_->Forward(bias_bottom_vec_, top);
  }

}

template <typename Dtype>
void AccScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(AccScaleLayer);
REGISTER_LAYER_CLASS(AccScale);

}  // namespace caffe
