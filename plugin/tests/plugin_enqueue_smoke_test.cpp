#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <vector>

#include "alignbev_plugin.h"
#include "bevpool_plugin.h"
#include "gatherbev_plugin.h"

namespace {

bool NearlyEqual(float a, float b, float eps = 1e-5F) { return std::fabs(a - b) <= eps; }

nvinfer1::PluginTensorDesc MakeTensorDesc(nvinfer1::DataType type, int nb_dims, const std::vector<int>& dims) {
  nvinfer1::PluginTensorDesc d{};
  d.type = type;
  d.format = nvinfer1::TensorFormat::kLINEAR;
  d.dims.nbDims = nb_dims;
  for (int i = 0; i < nb_dims; ++i) {
    d.dims.d[i] = dims.at(i);
  }
  return d;
}

bool RunBevPoolPluginEnqueueTest() {
  constexpr int channels = 2;
  constexpr int map_size = 4;
  constexpr int n_intervals = 3;

  std::vector<float> h_depth = {0.1F, 0.5F, 1.0F, 0.2F, 0.8F};
  std::vector<float> h_feat = {
      1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F, 5.0F, 50.0F,
  };
  std::vector<int> h_ranks_depth = {0, 1, 2, 3, 4};
  std::vector<int> h_ranks_feat = {0, 1, 2, 3, 4};
  std::vector<int> h_ranks_bev = {0, 0, 2, 2, 3};
  std::vector<int> h_interval_starts = {0, 2, 4};
  std::vector<int> h_interval_lengths = {2, 2, 1};

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

  float *d_bev_template = nullptr, *d_depth = nullptr, *d_feat = nullptr, *d_out = nullptr;
  int *d_ranks_depth = nullptr, *d_ranks_feat = nullptr, *d_ranks_bev = nullptr;
  int *d_interval_starts = nullptr, *d_interval_lengths = nullptr;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  cudaMalloc(reinterpret_cast<void**>(&d_bev_template), channels * map_size * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_depth), h_depth.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_feat), h_feat.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_out), channels * map_size * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_ranks_depth), h_ranks_depth.size() * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&d_ranks_feat), h_ranks_feat.size() * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&d_ranks_bev), h_ranks_bev.size() * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&d_interval_starts), h_interval_starts.size() * sizeof(int));
  cudaMalloc(reinterpret_cast<void**>(&d_interval_lengths), h_interval_lengths.size() * sizeof(int));

  cudaMemsetAsync(d_bev_template, 0, channels * map_size * sizeof(float), stream);
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

  tio::BevPoolPlugin plugin("bevpool_enqueue_smoke");
  nvinfer1::PluginTensorDesc in_desc[8] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {1, channels, 2, 2}),    // template
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 1, {static_cast<int>(h_depth.size())}),  // depth
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 2, {5, channels}),           // feat
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {5}),                     // ranks_depth
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {5}),                     // ranks_feat
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {5}),                     // ranks_bev
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {3}),                     // interval_starts
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {3}),                     // interval_lengths
  };
  nvinfer1::PluginTensorDesc out_desc[1] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {1, channels, 2, 2}),
  };
  const void* inputs[8] = {d_bev_template,   d_depth,     d_feat,        d_ranks_depth,
                           d_ranks_feat,     d_ranks_bev, d_interval_starts, d_interval_lengths};
  void* outputs[1] = {d_out};
  const int rc = plugin.enqueue(in_desc, out_desc, inputs, outputs, nullptr, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(channels * map_size, 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok = (rc == 0);
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_ref[i])) {
      std::cerr << "[BevPoolPlugin enqueue] mismatch at " << i << ": got " << h_out[i] << ", expect " << h_ref[i]
                << "\n";
      ok = false;
    }
  }

  cudaFree(d_bev_template);
  cudaFree(d_depth);
  cudaFree(d_feat);
  cudaFree(d_out);
  cudaFree(d_ranks_depth);
  cudaFree(d_ranks_feat);
  cudaFree(d_ranks_bev);
  cudaFree(d_interval_starts);
  cudaFree(d_interval_lengths);
  cudaStreamDestroy(stream);
  return ok;
}

bool RunAlignPluginEnqueueTest() {
  constexpr int adj = 2;
  constexpr int c = 1;
  constexpr int h = 3;
  constexpr int w = 3;
  constexpr int map = h * w;

  std::vector<float> h_in(adj * c * map, 0.0F);
  for (int i = 0; i < map; ++i) {
    h_in[i] = static_cast<float>(i);
    h_in[map + i] = static_cast<float>(100 + i);
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
  cudaMalloc(reinterpret_cast<void**>(&d_in), h_in.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_t), h_t.size() * sizeof(float));
  cudaMalloc(reinterpret_cast<void**>(&d_out), h_in.size() * sizeof(float));
  cudaMemcpyAsync(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_t, h_t.data(), h_t.size() * sizeof(float), cudaMemcpyHostToDevice, stream);

  tio::AlignBevPlugin plugin("alignbev_enqueue_smoke");
  nvinfer1::PluginTensorDesc in_desc[2] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {adj, c, h, w}),
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 2, {adj, 9}),
  };
  nvinfer1::PluginTensorDesc out_desc[1] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {adj, c, h, w}),
  };
  const void* inputs[2] = {d_in, d_t};
  void* outputs[1] = {d_out};
  const int rc = plugin.enqueue(in_desc, out_desc, inputs, outputs, nullptr, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(h_in.size(), 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok = (rc == 0);
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_in[i])) {
      std::cerr << "[AlignBevPlugin enqueue] mismatch at " << i << ": got " << h_out[i] << ", expect " << h_in[i]
                << "\n";
      ok = false;
    }
  }

  cudaFree(d_in);
  cudaFree(d_t);
  cudaFree(d_out);
  cudaStreamDestroy(stream);
  return ok;
}

bool RunGatherPluginEnqueueTest() {
  constexpr int b = 1;
  constexpr int adj = 2;
  constexpr int c = 1;
  constexpr int h = 2;
  constexpr int w = 2;
  constexpr int map = h * w;

  std::vector<float> h_adj = {
      10, 11, 12, 13,  // adj-0
      20, 21, 22, 23,  // adj-1
  };
  std::vector<float> h_curr = {1, 2, 3, 4};
  std::vector<int> h_flag = {1};
  std::vector<float> h_ref = {
      1, 2, 3, 4,      // curr
      10, 11, 12, 13,  // adj0
      20, 21, 22, 23,  // adj1
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

  tio::GatherBevPlugin plugin("gatherbev_enqueue_smoke");
  nvinfer1::PluginTensorDesc in_desc[3] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 5, {b, adj, c, h, w}),
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {b, c, h, w}),
      MakeTensorDesc(nvinfer1::DataType::kINT32, 1, {b}),
  };
  nvinfer1::PluginTensorDesc out_desc[1] = {
      MakeTensorDesc(nvinfer1::DataType::kFLOAT, 4, {b, (adj + 1) * c, h, w}),
  };
  const void* inputs[3] = {d_adj, d_curr, d_flag};
  void* outputs[1] = {d_out};
  const int rc = plugin.enqueue(in_desc, out_desc, inputs, outputs, nullptr, stream);
  cudaStreamSynchronize(stream);

  std::vector<float> h_out(h_ref.size(), 0.0F);
  cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bool ok = (rc == 0);
  for (std::size_t i = 0; i < h_out.size(); ++i) {
    if (!NearlyEqual(h_out[i], h_ref[i])) {
      std::cerr << "[GatherBevPlugin enqueue] mismatch at " << i << ": got " << h_out[i]
                << ", expect " << h_ref[i] << "\n";
      ok = false;
    }
  }

  cudaFree(d_adj);
  cudaFree(d_curr);
  cudaFree(d_out);
  cudaFree(d_flag);
  cudaStreamDestroy(stream);
  return ok;
}

}  // namespace

int main() {
  const bool bevpool_ok = RunBevPoolPluginEnqueueTest();
  const bool align_ok = RunAlignPluginEnqueueTest();
  const bool gather_ok = RunGatherPluginEnqueueTest();
  if (!bevpool_ok || !align_ok || !gather_ok) {
    std::cerr << "Plugin enqueue smoke test FAILED\n";
    return 1;
  }
  std::cout << "Plugin enqueue smoke test PASSED\n";
  return 0;
}
