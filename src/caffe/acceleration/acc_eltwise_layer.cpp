#include <cfloat>
#include <vector>

#include "caffe/acceleration/acc_eltwise_layer.hpp"
#include "caffe/acceleration/acc_util.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AccEltwiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param().eltwise_param().coeff_size() == 0
      || this->layer_param().eltwise_param().coeff_size() == bottom.size()) <<
      "Eltwise Layer takes one coefficient per bottom blob.";
  CHECK(!(this->layer_param().eltwise_param().operation()
      == EltwiseParameter_EltwiseOp_PROD
      && this->layer_param().eltwise_param().coeff_size())) <<
      "Eltwise layer only takes coefficients for summation.";
  op_ = this->layer_param_.eltwise_param().operation();
  // Blob-wise coefficients for the elementwise operation.
  coeffs_ = vector<Dtype>(bottom.size(), 1);
  if (this->layer_param().eltwise_param().coeff_size()) {
    for (int i = 0; i < bottom.size(); ++i) {
      coeffs_[i] = this->layer_param().eltwise_param().coeff(i);
      CHECK_EQ(coeffs_[i], 1);
    }
  }
  //stable_prod_grad_ = this->layer_param_.eltwise_param().stable_prod_grad();
  CHECK_EQ(bottom.size(), 2);
  CHECK(this->layer_param_.eltwise_param().operation() !=
    EltwiseParameter_EltwiseOp_MAX);
}

template <typename Dtype>
void AccEltwiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[1]->num(), bottom[0]->num());
  CHECK_EQ(bottom[1]->num(), 1);
  CHECK_EQ(bottom[1]->height(), bottom[0]->height());
  CHECK_EQ(bottom[1]->width(), bottom[0]->width());
  CHECK_LE(bottom[1]->channels(), bottom[0]->channels());
  top[0]->ReshapeLike(*bottom[0]);
  // If max operation, we will initialize the vector index part.
  //if (this->layer_param_.eltwise_param().operation() ==
  //    AccEltwiseParameter_EltwiseOp_MAX && top.size() == 1) {
  //  max_idx_.Reshape(bottom[0]->shape());
  //}
}

template <typename Dtype>
void AccEltwiseLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const int count_a = bottom[0]->count();
  const int count_b = bottom[1]->count();
  const int count_prod = bottom[0]->height() * bottom[0]->width();

  Dtype* top_data = top[0]->mutable_cpu_data();
  switch (op_) {
  case EltwiseParameter_EltwiseOp_PROD:
    if (bottom[0]->channels() == 1) {
      CHECK_EQ(bottom[1]->channels(), 1);
      caffe_mul(count_a, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    } else {
      if (bottom[1]->channels() == 1) {
        caffe_copy(count_a, bottom[0]->cpu_data(), top_data);
        top_data += count_a - count_prod;
        const Dtype* A = bottom[0]->cpu_data() + count_a - count_prod;
        const Dtype* B = bottom[1]->cpu_data();
        for (int i = 0; i < count_prod; i++) {
          top_data[i] = B[i] == Dtype(0) ? Dtype(1) : A[i] * B[i];
        }
      } else {
        CHECK_EQ(count_a, count_b);
        caffe_set(count_a, Dtype(0), top_data);
        top_data += count_a - count_prod;
        caffe_mul(count_prod, bottom[0]->cpu_data()+count_a-count_prod,
             bottom[1]->cpu_data()+count_b-count_prod, top_data);

        Dtype* top_base_data = top[0]->mutable_cpu_data();
        const Dtype* base_A = bottom[0]->cpu_data();
        const Dtype* base_B = bottom[1]->cpu_data();
        for (int channel = bottom[0]->channels()-1; channel; channel--) {
          const Dtype* A = bottom[0]->cpu_data() + count_a - count_prod;
          const Dtype* B = bottom[1]->cpu_data() + count_a - count_prod;
          for (int i = 0; i < count_prod; i++) {
            if(top_data[i] != Dtype(0)) {
              *(top_base_data++) = (*base_A) * (*base_B);
            }
            if(A[i] != Dtype(0))
              base_A++;
            if(B[i] != Dtype(0))
              base_B++;
          }
        }
      }
    }
    break;
  case EltwiseParameter_EltwiseOp_SUM:
    if (bottom[0]->channels() == 1) {
      CHECK_EQ(bottom[1]->channels(), 1);
      caffe_add(count_a, bottom[0]->cpu_data(), bottom[1]->cpu_data(), top_data);
    } else {
      if (bottom[1]->channels() == 1) {
        recovery_from_sparse(bottom[0]);
        caffe_copy(count_a, bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
        caffe_add(count_prod, bottom[0]->cpu_data(), bottom[1]->cpu_data(), 
            top[0]->mutable_cpu_data());
      } else {
        recovery_from_sparse(bottom[0]);
        recovery_from_sparse(bottom[1]);
        caffe_add(count_a-count_prod, bottom[0]->cpu_data(), 
            bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
        caffe_set(count_prod, Dtype(0), top[0]->mutable_cpu_data() + count_a - count_prod);
      }
    }
    break;
  case EltwiseParameter_EltwiseOp_MAX:
    // Initialize
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown elementwise operation.";
  }
}

template <typename Dtype>
void AccEltwiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(AccEltwiseLayer);
REGISTER_LAYER_CLASS(AccEltwise);

}  // namespace caffe
