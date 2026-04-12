#include "gatherbev_plugin.h"

namespace tio {

namespace {

__global__ void GatherFeatKernel(int nthreads, int adj_num, int channels, int map_size, const float* adj_feat,
                                 const float* curr_feat, const int* flags, float* out_feat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nthreads) {
    return;
  }
  const int b = idx / ((adj_num + 1) * map_size);
  const int n = (idx / map_size) % (adj_num + 1);
  const int m = idx % map_size;
  const int start = b * (adj_num + 1) * channels * map_size + n * channels * map_size + m;
  const int end = start + channels * map_size;
  for (int i = start, c = 0; i < end; i += map_size, ++c) {
    if (flags[b] == 0 || n == 0) {
      out_feat[i] = curr_feat[b * channels * map_size + c * map_size + m];
    } else {
      out_feat[i] = adj_feat[i - channels * map_size];
    }
  }
}

}  // namespace

void LaunchGatherBevKernel(const float* adj_feat, const float* curr_feat, const int* flags, float* out_feat, int b,
                           int adj_num, int channels, int map_size, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int nthreads = b * (adj_num + 1) * map_size;
  const int blocks = (nthreads + kThreads - 1) / kThreads;
  GatherFeatKernel<<<blocks, kThreads, 0, stream>>>(nthreads, adj_num, channels, map_size, adj_feat, curr_feat,
                                                     flags, out_feat);
}

}  // namespace tio
