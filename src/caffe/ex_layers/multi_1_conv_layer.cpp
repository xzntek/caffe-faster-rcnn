#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/ex_layers/multi_1_conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  CHECK_EQ(this->layer_param_.convolution_param().group(), 1);
  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ ;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ ;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);

  // For bottom size == 1 || bottom size == 2
  CHECK(bottom.size() == 1 || bottom.size() == 2);
  CHECK(!force_nd_im2col_);
  CHECK_EQ(num_spatial_axes_, 2);
  CHECK_EQ(this->layer_param_.convolution_param().dilation_size(), 0);
}

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  //for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
  //  CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
  //      << "All inputs must have the same shape.";
  //}
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ ;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ );
  for (int i = 0; i < num_spatial_axes_; ++i) {
    col_buffer_shape_.push_back(output_shape_[i]);
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
  CHECK_EQ(this->num_, 1);
  if (bottom.size() == 2) {
    CHECK_EQ(bottom[1]->num(), top[0]->num());
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), top[0]->height());
    CHECK_EQ(bottom[1]->width(), top[0]->width());
    top_buffer_.ReshapeLike(*top[0]);
  }
}

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = kernel_shape_data[i];
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::im2col_cpu_(const Dtype* data_im, Dtype* data_col) {
  const int channels = conv_in_channels_; // input channels, input height and width
  const int height = conv_input_shape_.cpu_data()[1];
  const int width = conv_input_shape_.cpu_data()[2];
  const int kernel_h = kernel_shape_.cpu_data()[0];
  const int kernel_w = kernel_shape_.cpu_data()[1];
  const int pad_h = pad_.cpu_data()[0];
  const int pad_w = pad_.cpu_data()[1];
  const int stride_h = stride_.cpu_data()[0];
  const int stride_w = stride_.cpu_data()[1];
  
  const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row ;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col ;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::im2col_cpu_P(const Dtype* data_im, Dtype* data_col, const Dtype* prod) {
  const int channels = conv_in_channels_; // input channels, input height and width
  const int height = conv_input_shape_.cpu_data()[1];
  const int width = conv_input_shape_.cpu_data()[2];
  const int kernel_h = kernel_shape_.cpu_data()[0];
  const int kernel_w = kernel_shape_.cpu_data()[1];
  const int pad_h = pad_.cpu_data()[0];
  const int pad_w = pad_.cpu_data()[1];
  const int stride_h = stride_.cpu_data()[0];
  const int stride_w = stride_.cpu_data()[1];
  
  const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row ;

        const Dtype *prod_ = prod;
        for (int output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              if (*(prod_++) != 0)
                  *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col ;
            for (int output_col = output_w; output_col; output_col--) {
              if (*(prod_++) != 0) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

template <typename Dtype>
void Multi1ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom.size() == 1) {

    Forward_Full(bottom[0], top[0]);

  } else if (bottom.size() == 2) {

    const int bottom1_count = bottom[1]->count();
    int sparsity = CalculateSparsity(bottom1_count, bottom[1]->cpu_data());
    DLOG(INFO) << "===>>> " << this->layer_param().name() << ", sparsity : " << sparsity;
    if (sparsity == 0 || is_1x1_) {
      Forward_Full(bottom[0], top[0]);
      Dtype* top_data = top[0]->mutable_cpu_data();
      for (int i = 0; i < bottom[0]->channels(); i++) {
        caffe_mul(bottom1_count, top_data, bottom[1]->cpu_data(), top_data);
        top_data += bottom[1]->count();
      }
    } else if (sparsity == bottom1_count) {

      caffe_set(top[0]->count(), Dtype(0), top[0]->mutable_cpu_data());

    } else {

      Forward_Second(bottom[0], &top_buffer_, bottom[1], bottom1_count - sparsity);
      // Recover
      const int top_dim = conv_out_spatial_dim_;
      Dtype* top_data = top[0]->mutable_cpu_data();
      const Dtype* prod_data = top_buffer_.cpu_data();
      for (int channel = conv_out_channels_; channel; channel--) {
        const Dtype* prod_ = bottom[1]->cpu_data();
        for (int x = top_dim; x; x--) {
          if (*prod_ == 0) {
            *(top_data++) = 0;
          } else {
            *(top_data++) = *(prod_data++) * (*prod_);
          }
          prod_++;
        }
      }
    }

  } else {
    LOG(FATAL) << "Unsopport bottom size";
  }
}

INSTANTIATE_CLASS(Multi1ConvolutionLayer);
REGISTER_LAYER_CLASS(Multi1Convolution);

}  // namespace caffe
