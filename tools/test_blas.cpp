#include <iostream>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/ex_layers/math_blas.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

int main(int argc, char** argv) {
    FLAGS_alsologtostderr = 1;
    ::google::InitGoogleLogging(argv[0]);
    //caffe::GlobalInit(&argc, &argv);

    Sparse_Matrix sparse;

    double sparse_ratio = 0;
    /*
    do{
        LOG(INFO) << "Input [Sparse Ratio, M, N]  (negative will end the process) : ";
        int M , N ;
        std::cin >> sparse_ratio >> M >> N;
        if( M < 0 || N < 0 ) break;
        sparse.TestMVProduct(M, N, sparse_ratio);
    }while( sparse_ratio>=0 && sparse_ratio <=1 );
    */
    
    do{
        LOG(INFO) << "Input [Spars Ratio[0], M, K, N]  (negative will end the process) : ";
        LOG(INFO) << "Test MM,  C = alpha * A * B + beta *C ,  A is m-by-k, B is k-by-n, C is m-by-n";
        int M, N, K;
        std::cin >> sparse_ratio >> M >> K >> N;
        if( N < 0 ) break;
        sparse.TestMMProduct(M, K, N, sparse_ratio);
    }while( sparse_ratio>=0 && sparse_ratio <=1 );

    return 0;
}
