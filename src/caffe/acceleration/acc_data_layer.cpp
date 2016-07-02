#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/acceleration/acc_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
void AccDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  CHECK_EQ(this->phase_, TEST);
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();

  const int batch_size = this->layer_param_.data_param().batch_size();
  CHECK_EQ(batch_size, 1);
  // Read a data point, and use it to initialize the top blob.
  Datum& datum = *(reader_.full().peek());

  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = 1;

  DLOG(INFO) << "Original channels : " << top_shape[1] << ", Pad To : " << top_shape[1]+1;
  top_shape[1]++; //All Ones ^_^

  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void AccDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  //const int batch_size = this->layer_param_.data_param().batch_size();
  //Datum& datum = *(reader_.full().peek());
  // Use data_transformer to infer the expected blob shape from datum.
  //vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  //this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  //top_shape[0] = batch_size;
  //top[0]->Reshape(top_shape);

  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = top[1]->mutable_cpu_data();
  }
  timer.Start();
  // get a datum
  Datum& datum = *(reader_.full().pop("Waiting for data"));
  read_time += timer.MicroSeconds();
  timer.Start();
  // Apply data transformations (mirror, scale, crop...)
  this->transformed_data_.set_cpu_data(top_data);
  this->data_transformer_->Transform(datum, &(this->transformed_data_));
  CHECK_EQ(transformed_data_.num(), top[0]->num());
  CHECK_EQ(transformed_data_.channels()+1, top[0]->channels());
  CHECK_EQ(transformed_data_.height(), top[0]->height());
  CHECK_EQ(transformed_data_.width(), top[0]->width());

  const int dims = top[0]->height() * top[0]->width();
  caffe_set(dims, Dtype(1), top[0]->mutable_cpu_data() + transformed_data_.count());

  // Copy label.
  if (this->output_labels_) {
      top_label[0] = datum.label();
  }
  trans_time += timer.MicroSeconds();

  reader_.free().push(const_cast<Datum*>(&datum));
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AccDataLayer);
REGISTER_LAYER_CLASS(AccData);

}  // namespace caffe
