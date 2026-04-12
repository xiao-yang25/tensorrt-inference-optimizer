#pragma once

#include <cuda_runtime_api.h>

namespace tio {

void LaunchBevPoolV2(int channels, int n_intervals, int map_size, const float* depth, const float* feat,
                     const int* ranks_depth, const int* ranks_feat, const int* ranks_bev,
                     const int* interval_starts, const int* interval_lengths, float* out,
                     cudaStream_t stream);

}  // namespace tio
