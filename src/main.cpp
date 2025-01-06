#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include <nccl.h>  // NCCL header

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    std::cerr << "Cuda error " << __FILE__ << ":" << __LINE__ \
              << " '" << cudaGetErrorString(err) << "'\n";      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    std::cerr << "NCCL error " << __FILE__ << ":" << __LINE__ \
              << " '" << ncclGetErrorString(res) << "'\n";      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define PRINT_MATRIX(m, n, A) \
  do {                         \
    for (size_t i = 0; i < m; i++) { \
      for (size_t j = 0; j < n; j++) { \
        std::cout << A[i * n + j] << " "; \
      } \
      std::cout << std::endl; \
    } \
    std::cout << std::endl; \
  } while (0)


#define NUM_GPUS 2

namespace gemm {

  template <typename T>
  void launch_gemm_v4(size_t m, size_t n, size_t k, const T* alpha, const T *A,
                      size_t lda, const T *B, size_t ldb, const T* beta, T *C,
                      size_t ldc, cudaStream_t stream);


  void launch_cublas_gemm(size_t m, size_t n, size_t k, const float* alpha, const float *A,
                          size_t lda, const float *B, size_t ldb, const float* beta, float *C,
                          size_t ldc) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc);
    cublasDestroy(handle);
  }

}
bool compare(const float *A, const float *B, const size_t& size) {
  for (size_t i = 0; i < size; i++) {
    if (std::abs(A[i] - B[i]) > 1e-6) {
      return false;
    }
  }
  return true;
}

using T = float;

int main(int argc, char **argv) {
  size_t m = 2048;
  size_t n = 1024;
  size_t k = 512;

  T *A = new T[m * k];
  T *B = new T[k * n];
  T *C = new T[m * n];
  T *C1 = new T[m * n];

  // set random seed
  std::generate(A, A + m * k, []() { return (T)(rand() % 10); });
  std::generate(B, B + k * n, []() { return (T)(rand() % 10); });
  std::fill(C, C + m * n, 0.0f);  // 初始化C矩阵为0
  std::fill(C1, C1 + m * n, 0.0f);

  cudaSetDevice(0);
  T *dA, *dB, *dC, *dC1;
  cudaMalloc(&dA, m * k * sizeof(T));
  cudaMalloc(&dB, k * n * sizeof(T));
  cudaMalloc(&dC, m * n * sizeof(T));
  cudaMalloc(&dC1, m * n * sizeof(T));

  cudaMemcpy(dA, A, m * k * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, k * n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C, m * n * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC1, C1, m * n * sizeof(T), cudaMemcpyHostToDevice);

  T alpha = 1.f;
  T beta = 0.f;

  size_t lda = k;
  size_t ldb = n;
  size_t ldc = n;

  // cublas gemm 
  gemm::launch_cublas_gemm(m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC1, ldc);
  cudaMemcpy(C1, dC1, m * n * sizeof(T), cudaMemcpyDeviceToHost);
  
  // cudaStream_t stream;
  // cudaStreamCreate(&stream);

  int num_gpus = NUM_GPUS; // 设置GPU数量
  int devs[num_gpus] = {0, 1};  // 假设我们有两个GPU
  ncclComm_t comms[num_gpus];
  cudaStream_t streams[num_gpus];

  size_t sub_m = m / num_gpus; // 将矩阵A按行分块

  // 初始化NCCL通信器
  NCCLCHECK(ncclCommInitAll(comms, num_gpus, devs));

  // 为每个GPU分配内存
  T* dA_gpu[num_gpus], *dB_gpu[num_gpus], *dC_gpu[num_gpus];
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    cudaMalloc(&dA_gpu[i], sub_m * k * sizeof(T));  // 分配A的子矩阵
    cudaMalloc(&dB_gpu[i], k * n * sizeof(T));      // 分配B
    cudaMalloc(&dC_gpu[i], sub_m * n * sizeof(T));  
    cudaStreamCreate(&streams[i]);
  }

  // 将A矩阵分块并复制到每个GPU， B不分块
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    cudaMemcpyAsync(dA_gpu[i], A + i * sub_m * k, sub_m * k * sizeof(T), cudaMemcpyHostToDevice, streams[i]);
    cudaMemcpyAsync(dB_gpu[i], B, k * n * sizeof(T), cudaMemcpyHostToDevice, streams[i]);
    //cudaMemcpyAsync(dC_gpu[i], C, m * n * sizeof(T), cudaMemcpyHostToDevice, streams[i]);
  }
  
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    cudaStreamSynchronize(streams[i]);
  }

  // 每个GPU执行自己的局部GEMM计算
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    gemm::launch_gemm_v4(sub_m, n, k, &alpha, dA_gpu[i], lda, dB_gpu[i], ldb, &beta, dC_gpu[i], ldc, streams[i]);
  }
  
  // NCCLCHECK(ncclGroupStart());
  //for (int i = 0; i < num_gpus; ++i) {
    // NCCLCHECK(ncclAllGather((const void*)dC_gpu[i], (void*)(dC_gpu[0]+ i * sub_m * n), sub_m * n, ncclFloat, comms[i], streams[i]));
    // NCCLCHECK(ncclReduce((const void*)dC_gpu[i], (void*)( dC + i * sub_m * n ), sub_m * n, ncclFloat, ncclSum, 0, comms[i], streams[i]));
  //}
  // for (int i = 0; i < num_gpus; ++i) {  // 从 1 开始，跳过 GPU 0
  //   NCCLCHECK(ncclRecv((void*)(dC+ i * sub_m * n), sub_m * n, ncclFloat, i, comms[0], streams[0]));  
  // }
  // for (int i = 0; i < num_gpus; ++i) {
  //   NCCLCHECK(ncclSend((const void*)dC_gpu[i], sub_m * n, ncclFloat, 0, comms[i], streams[i]));  
  // }
  // NCCLCHECK(ncclGroupEnd());  
  for (int i = 0; i < num_gpus; ++i) {
    cudaMemcpyAsync(dC + i * sub_m * n, dC_gpu[i], sub_m * n * sizeof(float), cudaMemcpyDeviceToDevice, streams[0]);
  }

  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    cudaStreamSynchronize(streams[i]);
  }

  // copy back C
  cudaSetDevice(devs[0]);
  cudaMemcpy(C, dC, m * n * sizeof(T), cudaMemcpyDeviceToHost);

  // // CUBLAS compare
  if (compare(C, C1, m * n)) {
    std::cout << "Success" << std::endl;
  } else {
    std::cout << "Failed" << std::endl;
  }

  // PRINT_MATRIX(m, k, A);
  // std::cout << "=========================" << std::endl;
  // PRINT_MATRIX(k, n, B);
  // std::cout << "=========================" << std::endl;
  // PRINT_MATRIX(m, n, C);
  // std::cout << "=========================" << std::endl;
  // PRINT_MATRIX(m, n, C1);


  // 清理资源
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(devs[i]);
    cudaFree(dA_gpu[i]);
    cudaFree(dB_gpu[i]);
    cudaFree(dC_gpu[i]);
    ncclCommDestroy(comms[i]);
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaFree(dC1);

  delete[] A;
  delete[] B;
  delete[] C;
  delete[] C1;

  return 0;
}
