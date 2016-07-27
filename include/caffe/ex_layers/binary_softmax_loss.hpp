#ifndef CAFFE_BINARY_SOFTMAX_LOSS_LAYER_HPP_
#define CAFFE_BINARY_SOFTMAX_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/softmax_loss_layer.hpp"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class BinarySoftmaxlossLayer : public LossLayer<Dtype> {
 public:
  explicit BinarySoftmaxlossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BinarySoftmaxloss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int cls_num;
  int batch_num;
  vector<shared_ptr<SoftmaxWithLossLayer<Dtype> > > softmaxloss_vecs_;
  vector<vector<Blob<Dtype>*> > top_vec_;
  vector<vector<Blob<Dtype>*> > bottom_vec_;
  vector<shared_ptr<Blob<Dtype> > > top_vec_shared_;
  vector<shared_ptr<Blob<Dtype> > > bottom_vec_shared_;
  vector<shared_ptr<Blob<Dtype> > > label_vec_shared_;
};

}  // namespace caffe

#endif  // CAFFE_BINARY_SOFTMAX_LOSS_LAYER_HPP_
