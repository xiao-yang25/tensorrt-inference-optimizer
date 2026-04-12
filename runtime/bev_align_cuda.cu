#include "bev_align_cuda.h"

namespace tio {

__global__ void ComputeSampleGridKernel(float* grid, const float* transform, int bev_w, int bev_h) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = bev_w * bev_h;
  if (idx >= total) {
    return;
  }
  const int x = idx / bev_h;
  const int y = idx % bev_h;
  const float src_x = transform[0] * y + transform[1] * x + transform[2];
  const float src_y = transform[3] * y + transform[4] * x + transform[5];
  grid[idx * 2 + 0] = src_x / (bev_w - 1.0F) * 2.0F - 1.0F;
  grid[idx * 2 + 1] = src_y / (bev_h - 1.0F) * 2.0F - 1.0F;
}

__device__ __forceinline__ float GridToPixel(float v, int size) {
  return ((v + 1.0F) / 2.0F) * (size - 1);
}

__global__ void GridSampleBilinearKernel(float* output, const float* input, const float* grid, int channels,
                                         int bev_w, int bev_h) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int map_size = bev_w * bev_h;
  const int total = channels * map_size;
  if (idx >= total) {
    return;
  }
  const int c = idx / map_size;
  const int p = idx % map_size;

  const float gx = GridToPixel(grid[p * 2 + 0], bev_w);
  const float gy = GridToPixel(grid[p * 2 + 1], bev_h);
  const int x0 = static_cast<int>(floorf(gx));
  const int y0 = static_cast<int>(floorf(gy));
  const int x1 = x0 + 1;
  const int y1 = y0 + 1;

  const float wx = gx - x0;
  const float wy = gy - y0;
  const float w00 = (1.0F - wx) * (1.0F - wy);
  const float w01 = wx * (1.0F - wy);
  const float w10 = (1.0F - wx) * wy;
  const float w11 = wx * wy;

  float val = 0.0F;
  const float* in_c = input + c * map_size;
  if (x0 >= 0 && x0 < bev_w && y0 >= 0 && y0 < bev_h) {
    val += in_c[x0 * bev_h + y0] * w00;
  }
  if (x1 >= 0 && x1 < bev_w && y0 >= 0 && y0 < bev_h) {
    val += in_c[x1 * bev_h + y0] * w01;
  }
  if (x0 >= 0 && x0 < bev_w && y1 >= 0 && y1 < bev_h) {
    val += in_c[x0 * bev_h + y1] * w10;
  }
  if (x1 >= 0 && x1 < bev_w && y1 >= 0 && y1 < bev_h) {
    val += in_c[x1 * bev_h + y1] * w11;
  }
  output[idx] = val;
}

void LaunchComputeSampleGrid(float* grid_dev, const float* transform_3x3_dev, int bev_w, int bev_h,
                             cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int total = bev_w * bev_h;
  const int blocks = (total + kThreads - 1) / kThreads;
  ComputeSampleGridKernel<<<blocks, kThreads, 0, stream>>>(grid_dev, transform_3x3_dev, bev_w, bev_h);
}

void LaunchGridSampleBilinear(float* output, const float* input, const float* grid, int channels, int bev_w,
                              int bev_h, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int total = channels * bev_w * bev_h;
  const int blocks = (total + kThreads - 1) / kThreads;
  GridSampleBilinearKernel<<<blocks, kThreads, 0, stream>>>(output, input, grid, channels, bev_w, bev_h);
}

}  // namespace tio
