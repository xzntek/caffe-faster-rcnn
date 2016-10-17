#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/signal_handler.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
DEFINE_string(gpu, "", 
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");

inline bool has_string(string A, string sub) {
  string::size_type idx = A.find( sub );
  return  idx != string::npos;
}

class Value {
public:
  Value(const string blob_name = ""){
    Init(blob_name);
  }
  void Init(const string blob_name_) {
    this->blob_name = blob_name_;
    values.clear();
  }
  void Add(const shared_ptr<Blob<float> > blob) {
    CHECK_EQ(blob->channels(), 1);
    for (int index = 0; index < blob->count(); index++) {
      values.push_back(blob->cpu_data()[index]);
    }
  }
  int Count(const float thresh = 0) {
    sort(values.begin(), values.end());
    int count = 0;
    for (size_t index = 0; index < values.size(); index++) {
      CHECK_GE(values[index], 0);
      if (values[index] <= thresh) {
        count ++;
      }
    }
    return count;
  }
  inline int Total() { return values.size(); }
  inline string Name() { return blob_name; }

private:
  vector<float> values;
  string blob_name;
};

int main(int argc, char** argv){
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --iterations   int     iterations to run");
  // Run tool or show usage.
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 || (FLAGS_gpu.size()==2&&FLAGS_gpu=="-1")) << "Can only support one gpu or none or -1(for cpu)";
  int gpu_id = -1;
  if( FLAGS_gpu.size() > 0 )
    gpu_id = boost::lexical_cast<int>(FLAGS_gpu);

  if (gpu_id >= 0) {
#ifndef CPU_ONLY
    caffe::Caffe::SetDevice(gpu_id);
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
#else
    LOG(FATAL) << "CPU ONLY MODEL, BUT PROVIDE GPU ID";
#endif
  } else {
    caffe::Caffe::set_mode(caffe::Caffe::CPU);
  }

  CHECK_GT(FLAGS_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  LOG(INFO) << "Running for " << FLAGS_iterations << " iterations.";

  const vector<shared_ptr<Layer<float> > > &layers = caffe_net.layers();
  vector<int> conv_layers;
  for (int index = 0; index < layers.size(); index++) {
    if(string(layers[index]->type()) == "Convolution") {
      conv_layers.push_back(index);
    }
  }
  LOG(INFO) << "Convolution Layer : " << conv_layers.size();

  const vector<string>& blob_names = caffe_net.blob_names();
  const vector<shared_ptr<Blob<float> > > &blobs = caffe_net.blobs();
  CHECK_EQ(blob_names.size(), blobs.size());


  vector<Value> Values;
  vector<int> single_blobs;
  const string suffix = "-acc-relu";
  for (int index = 0; index < blobs.size(); index++) {
    string name = blob_names[index];
    if (has_string(name, suffix)) {
      single_blobs.push_back(index);
      Values.push_back(Value(name));
    }
  }
  LOG(INFO) << suffix << " : " << single_blobs.size();

  vector<int> test_score_output_id;
  vector<float> test_score;
  float loss = 0;
  for (int i = 0; i < FLAGS_iterations; ++i) {
    float iter_loss;
    const vector<Blob<float>*>& result =
        caffe_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const float* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const float score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }   
        const std::string& output_name = caffe_net.blob_names()[
            caffe_net.output_blob_indices()[j]];
        LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
      }   
    }
    // Additional
    for (size_t index = 0; index < single_blobs.size(); index++) {
      const int idx = single_blobs[index];
      Values[index].Add(blobs[idx]);
    }
  }
  loss /= FLAGS_iterations;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = caffe_net.blob_names()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    const float loss_weight = caffe_net.blob_loss_weights()[
        caffe_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const float mean_score = test_score[i] / FLAGS_iterations;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
  }

  float threshes[] = {0, 0.001, 0.01, 0.1};
  const int count = sizeof(threshes) / sizeof(float);
  for (int i = 0; i < count; i++) {
    LOG(INFO) << "Thresh (" << std::setfill(' ') << std::setw(4) << threshes[i] << ")  ~~~~~~~~~~~~~~~~~~~~~~ Count values <= " << threshes[i];
    double Total = 0;
    double Count = 0;
    const float thresh = threshes[i];
    for (size_t index = 0; index < single_blobs.size(); index++) {
      const int total = Values[index].Total();
      const int count = Values[index].Count(thresh);
      const string name = Values[index].Name();
      double ratio = double(count) / total;
      LOG(INFO) << std::setfill(' ') << std::setw(10) << name << "\tratio : " << ratio << "\t" << count << " / " << total;
      Total += total / 100.f;
      Count += count / 100.f;
    }
    LOG(INFO) << std::setfill(' ') << std::setw(10) << "Count For Total Layer\tratio : " << Count / Total << "\t" << Count << " / " << Total << std::endl;
  }
  return 0;
}
