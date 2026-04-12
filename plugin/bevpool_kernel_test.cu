#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "bevpool_plugin.h"

namespace {

bool NearlyEqual(float a, float b, float eps = 1e-5F) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  // Tiny synthetic case for kernel correctness.
  constexpr int channels = 2;
  constexpr int map_size = 4;
  constexpr int n_intervals = 3;
  std::vector<float> h_depth = {0.1F, 0.5F, 1.0F, 0.2F, 0.8F};
  // feat is [point_count, channels]
  std::vector<float> h_feat = {
      1.0F, 10.0F,  // point 0
      2.0F, 20.0F,  // point 1
      3.0F, 30.0F,  // point 2
      4.0F, 40.0F,  // point 3
      5.0F, 50.0F,  // point 4
  };
  std::vector<int> h_ranks_depth = {0, 1, 2, 3, 4};
  std::vector<int> h_ranks_feat = {0, 1, 2, 3, 4};
  std::vector<int> h_ranks_bev = {0, 0, 2, 2, 3};
  std::vector<int> h_interval_starts = {0, 2, 4};
  std::vector<int> h_interval_lengths = {2, 2, 1};

  // CPU reference
  std::vector<float> h_ref(channels * map_size, 0.0F);
  for (int interval = 0; interval < n_intervals; ++interval) {
    const int start = h_interval_starts[interval];
    const int length = h_interval_lengths[interval];
    const int bev_rank = h_ranks_bev[start];
    for (int c = 0; c < channels; ++c) {
      float pooled = 0.0F;
      for (int i = 0; i < length; ++i) {
        const int p = start + i;
        pooled += h_depth[h_ranks_depth[p]] * h_feat[h_ranks_feat[p] * channels + c];
      }
      h_ref[c * map_size + bev_rank] = pooled;
    }
  }

  float *d_depth = nullptr, *d_feat = nullptr, *d_out = nullptr;
  int *d_ranks_depth = nullptr, *d_ranks_feat = nullptr, *d_ranks_bev = nullptr;
  int *d_interval_starts = nullptr, *d_interval_lengths = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  cudaMalloc(&d_depth, h_depth.size() * sizeof(float));
  cudaMalloc(&d_feat, h_feat.size() * sizeof(float));
  cudaMalloc(&d_out, channels * map_size * sizeof(float));
  cudaMalloc(&d_ranks_depth, h_ranks_depth.size() * sizeof(int));
  cudaMalloc(&d_ranks_feat, h_ranks_feat.size() * sizeof(int));
  cudaMalloc(&d_ranks_bev, h_ranks_bev.size() * sizeof(int));
  cudaMalloc(&d_interval_starts, h_interval_starts.size() * sizeof(int));
  cudaMalloc(&d_interval_lengths, h_interval_lengths.size() * sizeof(int));

  cudaMemcpyAsync(d_depth, h_depth.data(), h_depth.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_feat, h_feat.data(), h_feat.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_ranks_depth, h_ranks_depth.data(), h_ranks_depth.size() * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_ranks_feat, h_ranks_feat.data(), h_ranks_feat.size() * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_ranks_bev, h_ranks_bev.data(), h_ranks_bev.size() * sizeof(int), cudaMemcpyHostToDevice,
                  stream);
  cudaMemcpyAsync(d_interval_starts, h_interval_starts.data(), h_interval_starts.size() * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_interval_lengths, h_interval_lengths.data(), h_interval_lengths.size() * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(d_out, 0, channels * map_size * sizeof(float), stream);

  tio::LaunchBevPoolKernel(channels, n_intervals, map_size, d_depth, d_feat, d_ranks_depth, d_ranks_feat,
                           d_ranks_bev, d_interval_starts, d_interval_lengths, d_out, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(channels * map_size, 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_ref[i])) {
      std::cerr << "Mismatch at " << i << ": got " << h_out[i] << ", expect " << h_ref[i] << "\n";
      ok = false;
    }
  }

  cudaFree(d_depth);
  cudaFree(d_feat);
  cudaFree(d_out);
  cudaFree(d_ranks_depth);
  cudaFree(d_ranks_feat);
  cudaFree(d_ranks_bev);
  cudaFree(d_interval_starts);
  cudaFree(d_interval_lengths);
  cudaStreamDestroy(stream);

  if (!ok) {
    std::cerr << "BevPool kernel test FAILED\n";
    return 1;
  }
  std::cout << "BevPool kernel test PASSED\n";
  return 0;
}
