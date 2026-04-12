#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "gatherbev_plugin.h"

namespace {
bool NearlyEqual(float a, float b, float eps = 1e-5F) { return std::fabs(a - b) <= eps; }
}

int main() {
  constexpr int b = 1;
  constexpr int adj = 2;
  constexpr int c = 1;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int map = h * w;

  // adj_feat shape [1,2,1,2,2], each adj frame has distinct values.
  std::vector<float> h_adj = {
      10, 11, 12, 13,  // adj-0
      20, 21, 22, 23,  // adj-1
  };
  std::vector<float> h_curr = {1, 2, 3, 4};  // [1,1,2,2]
  std::vector<int> h_flag = {1};

  // out shape [1,3,1,2,2]: [curr, adj0, adj1]
  std::vector<float> h_ref = {
      1, 2, 3, 4,      // slot-0 from curr
      10, 11, 12, 13,  // slot-1 from adj0
      20, 21, 22, 23,  // slot-2 from adj1
  };

  float *d_adj = nullptr, *d_curr = nullptr, *d_out = nullptr;
  int* d_flag = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cudaMalloc(reinterpret_cast<void**>(&d_adj), h_adj.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_curr), h_curr.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_out), h_ref.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_flag), h_flag.size() * sizeof(int));

  cudaMemcpyAsync(d_adj, h_adj.data(), h_adj.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_curr, h_curr.data(), h_curr.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_flag, h_flag.data(), h_flag.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(d_out, 0, h_ref.size() * sizeof(float), stream);

  tio::LaunchGatherBevKernel(d_adj, d_curr, d_flag, d_out, b, adj, c, map, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(h_ref.size(), 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_ref[i])) {
      std::cerr << "Mismatch at " << i << ": got " << h_out[i] << ", expect " << h_ref[i] << "\n";
      ok = false;
    }
  }

  cudaFree(d_adj);
  cudaFree(d_curr);
  cudaFree(d_out);
  cudaFree(d_flag);
  cudaStreamDestroy(stream);
  if (!ok) {
    std::cerr << "GatherBev kernel test FAILED\n";
    return 1;
  }
  std::cout << "GatherBev kernel test PASSED\n";
  return 0;
}
