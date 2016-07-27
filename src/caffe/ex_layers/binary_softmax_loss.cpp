#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/ex_layers/binary_softmax_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BinarySoftmaxlossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LayerParameter softmaxloss_param = this->layer_param();
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->channels()%2, 0);
  batch_num = bottom[0]->num();
  this->cls_num = bottom[0]->channels() / 2;
  softmaxloss_vecs_.resize(this->cls_num);
  top_vec_shared_.resize(this->cls_num);
  bottom_vec_shared_.resize(this->cls_num);
  top_vec_.resize(this->cls_num);
  bottom_vec_.resize(this->cls_num);
  label_vec_shared_.resize(this->cls_num);
  for (int index = 0; index < this->cls_num; index++) {
    softmaxloss_vecs_[index].reset(new SoftmaxWithLossLayer<Dtype>(softmaxloss_param));
    bottom_vec_shared_[index].reset(new Blob<Dtype>(batch_num, 2, 1, 1));
    label_vec_shared_[index].reset(new Blob<Dtype>(batch_num, 1, 1, 1));

    bottom_vec_[index].push_back(bottom_vec_shared_[index].get());
    bottom_vec_[index].push_back(label_vec_shared_[index].get());

    top_vec_shared_[index].reset(new Blob<Dtype>(1, 1, 1, 1));
    top_vec_[index].push_back(top_vec_shared_[index].get());
    softmaxloss_vecs_[index]->LayerSetUp(bottom_vec_[index], top_vec_[index]);
  }
}

template <typename Dtype>
void BinarySoftmaxlossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[0]->channels(), 2*this->cls_num);
  CHECK_EQ(bottom[0]->num(), batch_num);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[1]->num(), bottom[1]->count());
  for (int index = 0; index < this->cls_num; index++) {
    CHECK_EQ(bottom[0]->num(), bottom_vec_[index][0]->num());
    softmaxloss_vecs_[index]->Reshape(bottom_vec_[index], top_vec_[index]);
    if (index > 0) {
      CHECK_EQ(bottom_vec_[index][0]->shape_string(), bottom_vec_[index-1][0]->shape_string());
      CHECK_EQ(bottom_vec_[index][1]->shape_string(), bottom_vec_[index-1][1]->shape_string());
    }
    CHECK_EQ(top_vec_[index][0]->count(), 1);
  }
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void BinarySoftmaxlossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  for (int index = 0; index < this->cls_num; index++) {
    Blob<Dtype>* blob = bottom_vec_[index][0];
    caffe_set(blob->count(), Dtype(0), blob->mutable_cpu_data());
    blob = bottom_vec_[index][1];
    caffe_set(blob->count(), Dtype(-1), blob->mutable_cpu_data());
  }
  const int channels = bottom[0]->channels();
  CHECK_EQ(2*cls_num, channels);
  const Dtype* bottom_data = bottom[0]->cpu_data();
  for (int index = 0; index < bottom[0]->num(); index++) {
    int cls = bottom[1]->cpu_data()[index];
    CHECK_GT(cls, -1);
    CHECK_LT(cls, 2*this->cls_num);
    Dtype* mid_data = bottom_vec_[cls/2][0]->mutable_cpu_data();
    Dtype* mid_labl = bottom_vec_[cls/2][1]->mutable_cpu_data();
    int cur_cls = cls % 2;
    mid_labl[ index ] = cur_cls;
    if (cur_cls == 1) cls --;
    mid_data[ index * 2 + 0 ] = bottom_data[ index*channels + cls + 0];
    mid_data[ index * 2 + 1 ] = bottom_data[ index*channels + cls + 1];
    CHECK_LT(index*channels + cls + 1, bottom[0]->count());
    CHECK_LT(index * 2 + 1, bottom_vec_[cls/2][0]->count());
  }
  for (int index = 0; index < this->cls_num; index++) {
    softmaxloss_vecs_[index]->Forward(bottom_vec_[index], top_vec_[index]);
  }
  Dtype* top_data = top[0]->mutable_cpu_data();
  top_data[0] = 0;
  for (int index = 0; index < this->cls_num; index++) {
    top_data[0] += top_vec_[index][0]->cpu_data()[0];
  }
}

template <typename Dtype>
void BinarySoftmaxlossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    for (int index = 0; index < this->cls_num; index++) {
      softmaxloss_vecs_[index]->Backward(bottom_vec_[index], propagate_down, top_vec_[index]);
    }
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int index = 0; index < bottom[0]->num(); index++) {
      int cls = bottom[1]->cpu_data()[index];
      const Dtype* mid_diff = bottom_vec_[cls/2][0]->cpu_diff();
      if (cls % 2 == 1) cls --;
      bottom_diff[index*2*cls_num+cls+0] = mid_diff[index * 2 + 0];
      bottom_diff[index*2*cls_num+cls+1] = mid_diff[index * 2 + 1];
      CHECK_LT(index*2*cls_num+cls+1, bottom[0]->count());
    }
  }
}

INSTANTIATE_CLASS(BinarySoftmaxlossLayer);
REGISTER_LAYER_CLASS(BinarySoftmaxloss);

}  // namespace caffe
