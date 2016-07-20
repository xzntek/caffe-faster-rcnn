#ifndef CAFFE_MULTI1_CONVOLUTION_LAYER_HPP_
#define CAFFE_MULTI1_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 * 输入两个blobs，第二个为点对点乘上去的．
 */
template <typename Dtype>
class Multi1ConvolutionLayer : public Layer<Dtype> {
 public:
  explicit Multi1ConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param), ZERO_(0) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
    NOT_IMPLEMENTED;
  }

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() ;

  virtual inline const char* type() const { return "Multi1Convolution"; }

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;
  const vector<int>* bottom_shape_;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
  }
  inline void Forward_Full(const Blob<Dtype>* bottom, Blob<Dtype>* top) {
    const Dtype* weights = this->blobs_[0]->cpu_data();
    const Dtype* bottom_data = bottom->cpu_data();
    Dtype* top_data = top->mutable_cpu_data();
    const Dtype* col_buff = bottom_data;
    if (!is_1x1_) {
      im2col_cpu_(bottom_data, col_buffer_.mutable_cpu_data());
      col_buff = col_buffer_.cpu_data();
    }
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
          conv_out_spatial_dim_, kernel_dim_,
          (Dtype)1., weights, col_buff,
          (Dtype)0., top_data);
    if (this->bias_term_) {
      const Dtype* bias = this->blobs_[1]->cpu_data();
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
            out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
            (Dtype)1., top_data);
    }
  }
  inline int CalculateSparsity(int count, const Dtype* data) {
    int zeros = 0;
    while(count--) {
      zeros += data[count] == 0;
    }
    return zeros;
  }
  void im2col_cpu_(const Dtype* data_im, Dtype* data_col);
  void im2col_cpu_P(const Dtype* data_im, Dtype* data_col, const Dtype* prod);

  int num_kernels_im2col_;
  int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  const Dtype ZERO_;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
