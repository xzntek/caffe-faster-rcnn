#ifndef CAFFE_ACC_RECOVER_LAYER_HPP_
#define CAFFE_ACC_RECOVER_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity @f$ y = \max(0, x) @f$.
 *        The simple max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class AccRecoverLayer : public Layer<Dtype> {
 public:
  /**
   * @param param provides AccRecoverParameter relu_param,
   *     with AccRecoverLayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   */
  explicit AccRecoverLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AccRecover"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
