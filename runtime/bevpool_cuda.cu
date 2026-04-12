#include "bevpool_cuda.h"

namespace tio {

__global__ void bev_pool_v2_kernel(int channels, int n_intervals, int map_size,
                                   const float* __restrict__ depth, const float* __restrict__ feat,
                                   const int* __restrict__ ranks_depth, const int* __restrict__ ranks_feat,
                                   const int* __restrict__ ranks_bev,
                                   const int* __restrict__ interval_starts,
                                   const int* __restrict__ interval_lengths, float* __restrict__ out) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int interval_index = idx / channels;
  const int channel_index = idx % channels;
  if (interval_index >= n_intervals) {
    return;
  }

  const int interval_start = interval_starts[interval_index];
  const int interval_length = interval_lengths[interval_index];
  float pooled = 0.0F;
  for (int i = 0; i < interval_length; ++i) {
    const int point_offset = interval_start + i;
    const float* cur_depth = depth + ranks_depth[point_offset];
    const float* cur_feat = feat + ranks_feat[point_offset] * channels + channel_index;
    pooled += (*cur_feat) * (*cur_depth);
  }

  const int bev_rank = ranks_bev[interval_start];
  float* cur_out = out + channel_index * map_size + bev_rank;
  *cur_out = pooled;
}

void LaunchBevPoolV2(int channels, int n_intervals, int map_size, const float* depth, const float* feat,
                     const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                     const int* interval_starts, const int* interval_lengths, float* out,
                     cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int total = n_intervals * channels;
  const int blocks = (total + kThreads - 1) / kThreads;
  bev_pool_v2_kernel<<<blocks, kThreads, 0, stream>>>(channels, n_intervals, map_size, depth, feat, ranks_depth,
                                                       ranks_feat, ranks_bev, interval_starts,
                                                       interval_lengths, out);
}

}  // namespace tio
