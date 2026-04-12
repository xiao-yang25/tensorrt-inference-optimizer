#pragma once

#include <cuda_runtime_api.h>

namespace tio {

void LaunchComputeSampleGrid(float* grid_dev, const float* transform_3x3_dev, int bev_w, int bev_h,
                             cudaStream_t stream);

void LaunchGridSampleBilinear(float* output, const float* input, const float* grid, int channels, int bev_w,
                              int bev_h, cudaStream_t stream);

}  // namespace tio
