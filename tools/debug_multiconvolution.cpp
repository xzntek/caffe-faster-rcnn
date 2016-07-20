#ifdef WITH_PYTHON_LAYER
#include "boost/python.hpp"
namespace bp = boost::python;
#endif

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
using caffe::map;
using caffe::Timer;
using caffe::vector;
using std::ostringstream;
using std::pair;
using std::make_pair;

DEFINE_string(gpu, "",
    "Optional; run in GPU mode on given device IDs separated by ','."
    "Use '-gpu all' to run on all available GPUs. The effective training "
    "batch size is multiplied by the number of devices.");
DEFINE_string(ok_model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(check_model, "",
    "The model definition protocol buffer text file.");
DEFINE_string(weights, "",
    "Optional; the pretrained weights to initialize finetuning, "
    "separated by ','. Cannot be set simultaneously with snapshot.");
DEFINE_int32(iterations, 50,
    "The number of iterations to run.");
/*
DEBUG.sh ::
gpu=-1
ok_model=examples/resnet20_bn_bn_Abn/cifar10_res20_trainval.proto
check_model=examples/resnet20_bn_bn_Abn/cifar10_res20_multi.proto 
weights=examples/resnet20_bn_bn_Abn/snapshot/cifar10_res20_iter_70000.caffemodel
iters=30
OMP_NUM_THREADS=1 ./build/tools/debug_multiconvolution --gpu $gpu --ok_model $ok_model \
    --check_model $check_model --weights $weights --iterations $iters 2>&1 | tee examples/resnet20_bn_bn_Abn/debug_$$.log
*/

vector<pair<string, string> > Get_Check_Blob() {
  vector<pair<string, string> > ans;
  ans.push_back(make_pair("data", "data"));
  ans.push_back(make_pair("init", "init"));
  ans.push_back(make_pair("res1.0-sum", "res1.0-sum"));
  ans.push_back(make_pair("res1.1-acc1", "res1.1-acc1"));
  ans.push_back(make_pair("res1.1-acc2", "res1.1-acc2"));
  ans.push_back(make_pair("res1.2-acc1", "res1.2-acc1"));
  ans.push_back(make_pair("res1.2-acc2", "res1.2-acc2"));
  ans.push_back(make_pair("res2.0-acc1", "res2.0-acc1"));
  ans.push_back(make_pair("res2.0-acc2", "res2.0-acc2"));
  ans.push_back(make_pair("res2.1-acc1", "res2.1-acc1"));
  ans.push_back(make_pair("res2.1-acc2", "res2.1-acc2"));
  ans.push_back(make_pair("res2.2-acc1", "res2.2-acc1"));
  ans.push_back(make_pair("res2.2-acc2", "res2.2-acc2"));
  ans.push_back(make_pair("res3.0-acc1", "res3.0-acc1"));
  ans.push_back(make_pair("res3.0-acc2", "res3.0-acc2"));
  ans.push_back(make_pair("res3.1-acc1", "res3.1-acc1"));
  ans.push_back(make_pair("res3.1-acc2", "res3.1-acc2"));
  ans.push_back(make_pair("res3.2-acc1", "res3.2-acc1"));
  ans.push_back(make_pair("res3.2-acc2", "res3.2-acc2"));
  return ans;
}

void ForwardCheck(Net<float> &caffe_net, Net<float> &check_net) {
  caffe_net.ForwardFrom(0);
  check_net.ForwardFrom(0);
  vector<pair<string, string> > equal_names = Get_Check_Blob();
  for (vector<pair<string, string> >::iterator it = equal_names.begin(); it != equal_names.end(); it++) {
    string ok_name = it->first;
    string check_name = it->second;
    CHECK(caffe_net.has_blob(ok_name));
    CHECK(check_net.has_blob(check_name));
    const shared_ptr<Blob<float> > ok_blob = caffe_net.blob_by_name(ok_name);
    const shared_ptr<Blob<float> > check_blob = check_net.blob_by_name(check_name);
    CHECK_EQ(ok_blob->num(), check_blob->num());
    CHECK_EQ(ok_blob->channels(), check_blob->channels());
    CHECK_EQ(ok_blob->height(), check_blob->height());
    CHECK_EQ(ok_blob->width(), check_blob->width());
    LOG(INFO) << "Check " << ok_name << " <--> " << check_name << " blobs : " << ok_blob->shape_string();
    const float* ok_data = ok_blob->cpu_data();
    const float* check_data = check_blob->cpu_data();
    for (int i = 0; i < ok_blob->count(); ++i) {
      CHECK_EQ(ok_data[i], check_data[i]);
    }
  }
}

int main(int argc, char** argv) {
  // Print output to stderr (while still logging).
  FLAGS_alsologtostderr = 1;
  // Set version
  gflags::SetVersionString(AS_STRING(CAFFE_VERSION));
  // Usage message.
  gflags::SetUsageMessage("command line brew\n"
      "usage: demo_frcnn_api <args>\n\n"
      "args:\n"
      "  --gpu          7       use 7-th gpu device, default is cpu model\n"
      "  --model[ok,check]        file    protocol buffer text file\n"
      "  --weights      file    Trained Model\n"
      "  --iterations   int     iterations to run");
  caffe::GlobalInit(&argc, &argv);
  CHECK( FLAGS_gpu.size() == 0 || FLAGS_gpu.size() == 1 || (FLAGS_gpu.size()==2&&FLAGS_gpu=="-1")) << "Can only support one gpu or none or -1(for cpu)";
  int gpu_id = -1; 
  if (FLAGS_gpu.size() > 0) 
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
  CHECK_GT(FLAGS_ok_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_check_model.size(), 0) << "Need a model definition to score.";
  CHECK_GT(FLAGS_weights.size(), 0) << "Need model weights to score.";
  

  // Instantiate the caffe net.
  Net<float> caffe_net(FLAGS_ok_model, caffe::TEST);
  caffe_net.CopyTrainedLayersFrom(FLAGS_weights);
  Net<float> check_net(FLAGS_check_model, caffe::TEST);
  check_net.CopyTrainedLayersFrom(FLAGS_weights);

  LOG(INFO) << "caffe_net : " << FLAGS_ok_model;
  LOG(INFO) << "check_net : " << FLAGS_check_model;

  float initial_loss;
  caffe_net.Forward(&initial_loss);
  LOG(INFO) << "Initial Correct Loss: " << initial_loss;
  check_net.Forward(&initial_loss);
  LOG(INFO) << "Initial Checked Loss: " << initial_loss;
  // Do a clean forward and backward pass, so that memory allocation are done
  // and future iterations will be more stable.
  LOG(INFO) << "Performing Forward";
  // Note that for the speed benchmark, we will assume that the network does
  // not take any input blobs.

  LOG(INFO) << "*** Benchmark begins ***";
  LOG(INFO) << "Testing for " << FLAGS_iterations << " iterations.";
  for (int j = 0; j < FLAGS_iterations; ++j) {
    LOG(INFO) << std::setfill(' ') << std::setw(5) << j + 1 << " th Forward and Check";
    ForwardCheck(caffe_net, check_net);
  }

  return 0;
}

