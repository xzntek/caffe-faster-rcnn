#ifndef CAFFE_ACC_DATA_LAYER_HPP_
#define CAFFE_ACC_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_reader.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class AccDataLayer : public Layer<Dtype> {
 public:
  explicit AccDataLayer(const LayerParameter& param)
      : Layer<Dtype>(param), transform_param_(param.transform_param()), reader_(param) {}
  virtual ~AccDataLayer(){}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // AccDataLayer uses DataReader instead for sharing for parallelism
  virtual inline bool ShareInParallel() const { return true; }
  virtual inline const char* type() const { return "AccData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:

  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  TransformationParameter transform_param_;
  Blob<Dtype> transformed_data_;
  bool output_labels_;
  DataReader reader_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
