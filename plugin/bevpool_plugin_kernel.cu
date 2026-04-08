#include "bevpool_plugin.h"

namespace tio {

namespace {

__global__ void BevPoolIdentityKernel(const float* input, float* output, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx];
  }
}

}  // namespace

void LaunchBevPoolIdentityKernel(const float* input, float* output, int element_count, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (element_count + kThreads - 1) / kThreads;
  BevPoolIdentityKernel<<<blocks, kThreads, 0, stream>>>(input, output, element_count);
}

}  // namespace tio
