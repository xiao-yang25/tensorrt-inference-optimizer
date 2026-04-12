#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "alignbev_plugin.h"

namespace {
bool NearlyEqual(float a, float b, float eps = 1e-5F) { return std::fabs(a - b) <= eps; }
}  // namespace

int main() {
  constexpr int adj = 2;
  constexpr int c = 1;
  constexpr int h = 3;
  constexpr int w = 3;
  constexpr int map = h * w;
  std::vector<float> h_in(adj * c * map, 0.0F);
  for (int i = 0; i < map; ++i) {
    h_in[i] = static_cast<float>(i);           // frame-0
    h_in[map + i] = static_cast<float>(100 + i);  // frame-1
  }
  std::vector<float> h_t(adj * 9, 0.0F);
  for (int i = 0; i < adj; ++i) {
    h_t[i * 9 + 0] = 1.0F;
    h_t[i * 9 + 4] = 1.0F;
    h_t[i * 9 + 8] = 1.0F;
  }

  float *d_in = nullptr, *d_t = nullptr, *d_out = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  cudaMalloc(&d_in, h_in.size() * sizeof(float));
  cudaMalloc(&d_t, h_t.size() * sizeof(float));
  cudaMalloc(&d_out, h_in.size() * sizeof(float));
  cudaMemcpyAsync(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_t, h_t.data(), h_t.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

  tio::LaunchAlignBevKernel(d_in, d_t, d_out, adj, c, h, w, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(h_in.size(), 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_in[i])) {
      std::cerr << "Mismatch at " << i << ": got " << h_out[i] << ", expect " << h_in[i] << "\n";
      ok = false;
    }
  }

  cudaFree(d_in);
  cudaFree(d_t);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
  if (!ok) {
    std::cerr << "AlignBev kernel test FAILED\n";
    return 1;
  }
  std::cout << "AlignBev kernel test PASSED\n";
  return 0;
}
