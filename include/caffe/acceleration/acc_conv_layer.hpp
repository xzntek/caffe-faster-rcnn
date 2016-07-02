#ifndef CAFFE_ACC_CONVOLUTION_LAYER_HPP_
#define CAFFE_ACC_CONVOLUTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"

namespace caffe {

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ConvolutionLayer and DeconvolutionLayer.
 */
template <typename Dtype>
class AccConvolutionLayer : public Layer<Dtype> {
 public:
  explicit AccConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }
  virtual inline const char* type() const { return "AccConvolution"; }

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the im2col if
  // we just called weight_cpu_gemm with the same input.

  /// @brief The spatial dimensions of the input.
  inline int input_shape(int i) {
    return (*bottom_shape_)[channel_axis_ + i];
  }
  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  // virtual inline bool reverse_dimensions() { return false; }
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape();

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
  int group_;
  int out_spatial_dim_;
  int weight_offset_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;
  bool force_nd_im2col_;

 private:
  inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
    return static_cast<unsigned>(a) < static_cast<unsigned>(b);
  }

  int PrepareProd(const vector<Blob<Dtype>*>& bottom);

  inline void conv_im2col_1x1(const Dtype* data_im, Dtype* col_buff, const int stride) {
    // stride = x, kernal = 1, pad = 0;
    CHECK(force_nd_im2col_==false);
    CHECK_EQ(num_spatial_axes_, 2);
    const int channels = conv_in_channels_;
    const int height = conv_input_shape_.cpu_data()[1];  
    const int width = conv_input_shape_.cpu_data()[2];
    const int output_h = (height - 1 ) / stride + 1;
    const int output_w = (width - 1 ) / stride + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      int input_row = 0 ;
      for (int output_rows = output_h; output_rows; output_rows--) {
        int input_col = 0 ;
        for (int output_col = output_w; output_col; output_col--) {
          *(col_buff++) = data_im[input_row * width + input_col];
          input_col += stride;
        }
        input_row += stride;
      }
    }

  }
  // wrap im2col/col2im so we don't have to remember the (long) argument lists
  //inline void conv_im2col_cpu(const Dtype* data_im, Dtype* col_buff, const Dtype* prod_data_) {
  inline void conv_im2col_cpu(const Dtype* data_im, Dtype* col_buff) {
    CHECK(force_nd_im2col_==false);
    CHECK_EQ(num_spatial_axes_, 2);
    const int channels = conv_in_channels_;
    const int height = conv_input_shape_.cpu_data()[1];  
    const int width = conv_input_shape_.cpu_data()[2];
    const int kernel_h = kernel_shape_.cpu_data()[0];
    const int kernel_w = kernel_shape_.cpu_data()[1];
    const int pad_h = pad_.cpu_data()[0];
    const int pad_w = pad_.cpu_data()[1];
    const int stride_h = stride_.cpu_data()[0];
    const int stride_w = stride_.cpu_data()[1];
    const int output_h = (height + 2 * pad_h - kernel_h ) / stride_h + 1;
    const int output_w = (width + 2 * pad_w - kernel_w ) / stride_w + 1;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row ;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(col_buff++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col ;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(col_buff++) = data_im[input_row * width + input_col];
                } else {
                  *(col_buff++) = 0;
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

  //int num_kernels_im2col_;
  //int num_kernels_col2im_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int kernel_dim_;
  int col_offset_;
  int output_offset_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;

  Blob<Dtype> Prod;
  Blob<int> Index;
  Blob<int> Stride_Index;
};

}  // namespace caffe

#endif  // CAFFE_BASE_CONVOLUTION_LAYER_HPP_
