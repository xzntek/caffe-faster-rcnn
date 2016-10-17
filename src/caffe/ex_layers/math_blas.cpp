#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/ex_layers/math_blas.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype> 
void DisplayMatrix(const Dtype *A, const int m, const int n){
    if( m*n >= 100 ){
        LOG(INFO) << "Too Many to display.";
        return;
    }
    for (int index = 0; index < m; ++index){
        std::ostringstream buffer;
        for(int jj = 0; jj < n; ++jj){
            int x = A[index*n+jj] * 100;
            buffer << std::setfill(' ') << std::setw(6) << x/100.f;            
        }
        LOG(INFO) << buffer.str(); 
    }
}
template void DisplayMatrix<float>(const float *A, const int m, const int n);
template void DisplayMatrix<int>(const int *A, const int m, const int n);
template void DisplayMatrix<double>(const double *A, const int m, const int n);

int GetRandomMatrix(float *A, const int m, const int n, const int seed = 0,const double zero_ratio = 0){
    caffe::rng_t RD(seed); 
    int count_nozeros = m * n;
    for (int index = 0; index < m*n; ++index){
        A[index] = (RD()%5000+1) / 1000.;
        if( RD()%1000 < zero_ratio*1000 ) A[index] = 0;
        if( A[index] == 0 ) count_nozeros --;
    }
    return count_nozeros;
}

void Sparse_Matrix::TestMVProduct(const int m, const int n, const double zero_ratio){
    /*
       The since mkl cannot be used, this will only test dense matrix test mv product
y := alpha*A*x + beta*y  or  y := alpha*A'*x + beta*y,
where:
alpha and beta are scalars,
x and y are vectors,
A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
*/
    CHECK( zero_ratio >= 0 && zero_ratio <= 1 );
    CHECK( m > 0 && n > 0 );
    float *A = new float[m*n];
    const int count_nozeros = GetRandomMatrix(A , m , n , 0 , zero_ratio);
    LOG(INFO) << "Random Set Matrix A (" << m << " , " << n << ") : " << count_nozeros*1.0/(m*n) << " (force) no-zeros elements" ;
    DisplayMatrix(A, m , n );
    const int k = n;
    const float alpha = 3.9;  // Specifies the scalar beta. 
    float *x = new float[k]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
    const float beta = 1.7;  // Specifies the scalar beta. 
    float *y = new float[m];
    GetRandomMatrix(x , k , 1 , 101);
    GetRandomMatrix(y , 1 , m , 209);
    LOG(INFO) << "m : " << m << "  , k : " << k;
    LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
    LOG(INFO) << "[MV-Product] Data Prepare Done,.";
    const int LOOP = 30;
    vector<double> DenseAVE;
    caffe::Timer _time; 
    for (int loop = 0; loop < LOOP; ++loop){
        // Force to calculate Y
        _time.Start();
        caffe_cpu_gemv(CblasNoTrans, m, n, alpha, A, x, beta, y);
        double dense_time = _time.MicroSeconds();
        DenseAVE.push_back( dense_time );
        LOG(INFO) << "Loop " << loop << "Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
    }
    sort(DenseAVE.begin(), DenseAVE.end());
    double ave = 0 ;
    int num = 0;
    for (int i = 4; i < LOOP - 4 ; i ++ ){
        ave += DenseAVE[i];
        num ++;
    }
    LOG(INFO) << "[MV] Dense Average Time: " << ave/num << " us";

    delete []A;
    delete []x;
    delete []y;
}

MM_Time Sparse_Matrix::TestMMProduct(const int m, const int k, const int n ,const double zero_ratio){
    /*
       The since mkl cannot be used, this will only test dense matrix test mm product
C := alpha*A*B + beta*C  or  C := alpha*A'*B + beta*C,
where:
alpha and beta are scalars,
B and C are dense matrices ,
A is an m-by-k sparse matrix in the CSR format, A' is the transpose of A.
*/
    CHECK( zero_ratio >= 0 && zero_ratio <= 1 ) << " not " << zero_ratio;
    //const int m = k + 10;
    CHECK( m > 0 && k > 0 );
    float *A = new float[m*k];
    const int count_nozeros = GetRandomMatrix(A , m , k , 0 , zero_ratio);
    LOG(INFO) << "Random Set Matrix A (" << m << " , " << k << ") : " << count_nozeros*1.0/(m*k) << " (force) no-zeros elements" ;
    DisplayMatrix(A, m , k );
    //void mkl_scsrmv(char *transa, int *m, int *k, float *alpha, char *matdescra, float *val, int *indx, int *pntrb, int *pntre, float *x, float *beta, float *y);
    //const int k = n;              //INTEGER. Number of columns of the matrix A.
    const float alpha = 2.0;            // REAL for mkl_scsrmv.
    //const int n = k + 1;
    float *B = new float[k*n]; // Array, DIMENSION at least k if transa = 'N' or 'n' and at least m otherwise. On entry, the array x must contain the vector x. 
    const float beta = 1.7;  // Specifies the scalar beta. 
    float *C = new float[m*n];
    GetRandomMatrix(B , k , n , 121);
    GetRandomMatrix(C , m , n , 207);
    LOG(INFO) << "m : " << m << "  , k : " << k << " , n : " << n;
    LOG(INFO) << "alpha : " << alpha << "  , beta : " << beta;
    //LOG(INFO) << "[MV-Product] Data Prepare Done";
    const int LOOP = 30;
    vector<double> DenseAVE;
    caffe::Timer _time; 
    for (int loop = 0; loop < LOOP; ++loop){
        // Force to calculate Y
        _time.Start();
        caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, B, beta, C);
        double dense_time = _time.MicroSeconds();
        //LOG(INFO) << "Loop " << loop << "[Dense : " << std::setfill(' ') << std::setw(5) << dense_time << "] us";
        DenseAVE.push_back(dense_time);
    }
    sort(DenseAVE.begin(), DenseAVE.end());
    LOG(INFO) << "FOR CHECK >>>>  m : " << m << " n : " << n << " k : " << k;
    double ave = 0 ;
    int num = 0;
    for (int i = 4; i < LOOP - 4 ; i ++ ){
        ave += DenseAVE[i];
        num ++;
    }
    LOG(INFO) << "Dense Average Time: " << ave/num << " us";
    delete []A;
    delete []B;
    delete []C;
    return MM_Time(m, n, k, ave/num);
}

}   // namespace caffe
