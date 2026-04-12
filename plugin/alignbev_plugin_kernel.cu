#include "alignbev_plugin.h"

#include <cmath>

namespace tio {

namespace {

__device__ __forceinline__ float sample_bilinear(const float* in_c, int bev_h, int bev_w, float x, float y) {
  const int x0 = static_cast<int>(floorf(x));
  const int y0 = static_cast<int>(floorf(y));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const float wx = x - x0;
  const float wy = y - y0;
  const float w00 = (1.0F - wx) * (1.0F - wy);
  const float w01 = wx * (1.0F - wy);
  const float w10 = (1.0F - wx) * wy;
  const float w11 = wx * wy;
  float out = 0.0F;
  if (x0 >= 0 && x0 < bev_w && y0 >= 0 && y0 < bev_h) out += in_c[y0 * bev_w + x0] * w00;
  if (x1 >= 0 && x1 < bev_w && y0 >= 0 && y0 < bev_h) out += in_c[y0 * bev_w + x1] * w01;
  if (x0 >= 0 && x0 < bev_w && y1 >= 0 && y1 < bev_h) out += in_c[y1 * bev_w + x0] * w10;
  if (x1 >= 0 && x1 < bev_w && y1 >= 0 && y1 < bev_h) out += in_c[y1 * bev_w + x1] * w11;
  return out;
}

__global__ void AlignBevKernel(const float* input, const float* transforms, float* output, int adj_num, int channels,
                               int bev_h, int bev_w) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int map_size = bev_h * bev_w;
  const int total = adj_num * channels * map_size;
  if (idx >= total) {
    return;
  }
  const int a = idx / (channels * map_size);
  const int rem = idx % (channels * map_size);
  const int c = rem / map_size;
  const int p = rem % map_size;
  const int y = p / bev_w;
  const int x = p % bev_w;

  const float* t = transforms + a * 9;
  const float src_x = t[0] * x + t[1] * y + t[2];
  const float src_y = t[3] * x + t[4] * y + t[5];

  const float* in_c = input + a * channels * map_size + c * map_size;
  output[idx] = sample_bilinear(in_c, bev_h, bev_w, src_x, src_y);
}

}  // namespace

void LaunchAlignBevKernel(const float* input, const float* transforms, float* output, int adj_num, int channels,
                          int bev_h, int bev_w, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int total = adj_num * channels * bev_h * bev_w;
  const int blocks = (total + kThreads - 1) / kThreads;
  AlignBevKernel<<<blocks, kThreads, 0, stream>>>(input, transforms, output, adj_num, channels, bev_h, bev_w);
}

}  // namespace tio
