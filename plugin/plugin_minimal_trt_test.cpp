#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "alignbev_plugin.h"
#include "bevpool_plugin.h"
#include "gatherbev_plugin.h"
#include "logger.h"

namespace {

bool NearlyEqual(float a, float b, float eps = 1e-5F) { return std::fabs(a - b) <= eps; }

std::int64_t Volume(const nvinfer1::Dims& dims) {
  std::int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= static_cast<std::int64_t>(dims.d[i]);
  }
  return v;
}

std::size_t ElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kINT32:
      return sizeof(int);
    case nvinfer1::DataType::kHALF:
      return sizeof(std::uint16_t);
    default:
      return 0;
  }
}

nvinfer1::Dims MakeDims(std::initializer_list<int> vals) {
  nvinfer1::Dims d{};
  d.nbDims = static_cast<int>(vals.size());
  int i = 0;
  for (int v : vals) {
    d.d[i++] = v;
  }
  return d;
}

void PrintMismatch(const std::string& tag, int idx, float got, float expect) {
  std::cerr << "[" << tag << "] mismatch at " << idx << ": got " << got << ", expect " << expect << "\n";
}

}  // namespace

int main() {
  tio::TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
  std::unique_ptr<nvinfer1::IPluginV2> gather_plugin(new tio::GatherBevPlugin("gather"));
  std::unique_ptr<nvinfer1::IPluginV2> align_plugin(new tio::AlignBevPlugin("align"));
  std::unique_ptr<nvinfer1::IPluginV2> bevpool_plugin(new tio::BevPoolPlugin("bevpool"));
  if (!gather_plugin || !align_plugin || !bevpool_plugin) {
    std::cerr << "Plugin createPlugin failed\n";
    return 1;
  }

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) {
    std::cerr << "createInferBuilder failed\n";
    return 1;
  }
  const auto flags = 1U << static_cast<unsigned>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!network || !config) {
    std::cerr << "createNetwork/config failed\n";
    return 1;
  }

  // Gather inputs.
  auto* t_g_adj = network->addInput("g_adj", nvinfer1::DataType::kFLOAT, MakeDims({1, 2, 1, 2, 2}));
  auto* t_g_curr = network->addInput("g_curr", nvinfer1::DataType::kFLOAT, MakeDims({1, 1, 2, 2}));
  auto* t_g_flag = network->addInput("g_flag", nvinfer1::DataType::kINT32, MakeDims({1}));
  nvinfer1::ITensor* gather_inputs[] = {t_g_adj, t_g_curr, t_g_flag};
  auto* gather_layer = network->addPluginV2(gather_inputs, 3, *gather_plugin);
  gather_layer->getOutput(0)->setName("out_gather");
  network->markOutput(*gather_layer->getOutput(0));

  // Align inputs.
  auto* t_a_feat = network->addInput("a_feat", nvinfer1::DataType::kFLOAT, MakeDims({2, 1, 3, 3}));
  auto* t_a_tf = network->addInput("a_tf", nvinfer1::DataType::kFLOAT, MakeDims({2, 9}));
  nvinfer1::ITensor* align_inputs[] = {t_a_feat, t_a_tf};
  auto* align_layer = network->addPluginV2(align_inputs, 2, *align_plugin);
  align_layer->getOutput(0)->setName("out_align");
  network->markOutput(*align_layer->getOutput(0));

  // BevPool inputs.
  auto* t_b_template = network->addInput("b_template", nvinfer1::DataType::kFLOAT, MakeDims({1, 2, 2, 2}));
  auto* t_b_depth = network->addInput("b_depth", nvinfer1::DataType::kFLOAT, MakeDims({5}));
  auto* t_b_feat = network->addInput("b_feat", nvinfer1::DataType::kFLOAT, MakeDims({5, 2}));
  auto* t_b_rd = network->addInput("b_ranks_depth", nvinfer1::DataType::kINT32, MakeDims({5}));
  auto* t_b_rf = network->addInput("b_ranks_feat", nvinfer1::DataType::kINT32, MakeDims({5}));
  auto* t_b_rb = network->addInput("b_ranks_bev", nvinfer1::DataType::kINT32, MakeDims({5}));
  auto* t_b_is = network->addInput("b_interval_starts", nvinfer1::DataType::kINT32, MakeDims({3}));
  auto* t_b_il = network->addInput("b_interval_lengths", nvinfer1::DataType::kINT32, MakeDims({3}));
  nvinfer1::ITensor* bevpool_inputs[] = {t_b_template, t_b_depth, t_b_feat, t_b_rd, t_b_rf, t_b_rb, t_b_is, t_b_il};
  auto* bevpool_layer = network->addPluginV2(bevpool_inputs, 8, *bevpool_plugin);
  bevpool_layer->getOutput(0)->setName("out_bevpool");
  network->markOutput(*bevpool_layer->getOutput(0));

  if (!t_g_adj || !t_g_curr || !t_g_flag || !t_a_feat || !t_a_tf || !t_b_template || !t_b_depth || !t_b_feat ||
      !t_b_rd || !t_b_rf || !t_b_rb || !t_b_is || !t_b_il || !gather_layer || !align_layer || !bevpool_layer) {
    std::cerr << "Network construction failed\n";
    return 1;
  }

  std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
  if (!serialized) {
    std::cerr << "buildSerializedNetwork failed\n";
    return 1;
  }

  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
  std::unique_ptr<nvinfer1::IExecutionContext> context(engine ? engine->createExecutionContext() : nullptr);
  if (!runtime || !engine || !context) {
    std::cerr << "Engine/runtime/context create failed\n";
    return 1;
  }

  const int nb = engine->getNbIOTensors();
  std::vector<void*> io_ptrs(nb, nullptr);
  std::vector<const char*> io_names(nb, nullptr);
  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    io_names[i] = name;
    const auto dims = engine->getTensorShape(name);
    const auto type = engine->getTensorDataType(name);
    const std::size_t bytes = static_cast<std::size_t>(Volume(dims)) * ElementSize(type);
    if (bytes == 0) {
      std::cerr << "Unsupported binding type or size at " << i << "\n";
      return 1;
    }
    cudaMalloc(&io_ptrs[i], bytes);
    cudaMemset(io_ptrs[i], 0, bytes);
    if (!context->setTensorAddress(name, io_ptrs[i])) {
      std::cerr << "setTensorAddress failed for " << name << "\n";
      return 1;
    }
  }

  // Host data and refs for gather.
  std::vector<float> h_g_adj = {10, 11, 12, 13, 20, 21, 22, 23};
  std::vector<float> h_g_curr = {1, 2, 3, 4};
  std::vector<int> h_g_flag = {1};
  std::vector<float> h_g_ref = {1, 2, 3, 4, 10, 11, 12, 13, 20, 21, 22, 23};

  // Host data and refs for align.
  std::vector<float> h_a_feat(2 * 1 * 3 * 3, 0.0F);
  for (int i = 0; i < 9; ++i) {
    h_a_feat[i] = static_cast<float>(i);
    h_a_feat[9 + i] = static_cast<float>(100 + i);
  }
  std::vector<float> h_a_tf(2 * 9, 0.0F);
  for (int i = 0; i < 2; ++i) {
    h_a_tf[i * 9 + 0] = 1.0F;
    h_a_tf[i * 9 + 4] = 1.0F;
    h_a_tf[i * 9 + 8] = 1.0F;
  }
  std::vector<float> h_a_ref = h_a_feat;

  // Host data and refs for bevpool.
  std::vector<float> h_b_template(1 * 2 * 2 * 2, 0.0F);
  std::vector<float> h_b_depth = {0.1F, 0.5F, 1.0F, 0.2F, 0.8F};
  std::vector<float> h_b_feat = {1.0F, 10.0F, 2.0F, 20.0F, 3.0F, 30.0F, 4.0F, 40.0F, 5.0F, 50.0F};
  std::vector<int> h_b_rd = {0, 1, 2, 3, 4};
  std::vector<int> h_b_rf = {0, 1, 2, 3, 4};
  std::vector<int> h_b_rb = {0, 0, 2, 2, 3};
  std::vector<int> h_b_is = {0, 2, 4};
  std::vector<int> h_b_il = {2, 2, 1};
  std::vector<float> h_b_ref(2 * 4, 0.0F);
  for (int interval = 0; interval < 3; ++interval) {
    const int start = h_b_is[interval];
    const int length = h_b_il[interval];
    const int bev_rank = h_b_rb[start];
    for (int c = 0; c < 2; ++c) {
      float pooled = 0.0F;
      for (int i = 0; i < length; ++i) {
        const int p = start + i;
        pooled += h_b_depth[h_b_rd[p]] * h_b_feat[h_b_rf[p] * 2 + c];
      }
      h_b_ref[c * 4 + bev_rank] = pooled;
    }
  }

  auto copy_input = [&](const char* name, const void* src, std::size_t bytes) {
    int idx = -1;
    for (int i = 0; i < nb; ++i) {
      if (std::string(io_names[i]) == name) {
        idx = i;
        break;
      }
    }
    if (idx < 0) {
      std::cerr << "Binding not found: " << name << "\n";
      return false;
    }
    return cudaMemcpy(io_ptrs[idx], src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  };

  if (!copy_input("g_adj", h_g_adj.data(), h_g_adj.size() * sizeof(float)) ||
      !copy_input("g_curr", h_g_curr.data(), h_g_curr.size() * sizeof(float)) ||
      !copy_input("g_flag", h_g_flag.data(), h_g_flag.size() * sizeof(int)) ||
      !copy_input("a_feat", h_a_feat.data(), h_a_feat.size() * sizeof(float)) ||
      !copy_input("a_tf", h_a_tf.data(), h_a_tf.size() * sizeof(float)) ||
      !copy_input("b_template", h_b_template.data(), h_b_template.size() * sizeof(float)) ||
      !copy_input("b_depth", h_b_depth.data(), h_b_depth.size() * sizeof(float)) ||
      !copy_input("b_feat", h_b_feat.data(), h_b_feat.size() * sizeof(float)) ||
      !copy_input("b_ranks_depth", h_b_rd.data(), h_b_rd.size() * sizeof(int)) ||
      !copy_input("b_ranks_feat", h_b_rf.data(), h_b_rf.size() * sizeof(int)) ||
      !copy_input("b_ranks_bev", h_b_rb.data(), h_b_rb.size() * sizeof(int)) ||
      !copy_input("b_interval_starts", h_b_is.data(), h_b_is.size() * sizeof(int)) ||
      !copy_input("b_interval_lengths", h_b_il.data(), h_b_il.size() * sizeof(int))) {
    std::cerr << "Input memcpy failed\n";
    return 1;
  }

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  if (!context->enqueueV3(stream)) {
    std::cerr << "enqueueV3 failed\n";
    return 1;
  }
  cudaStreamSynchronize(stream);

  std::vector<float> h_g_out(h_g_ref.size(), 0.0F);
  std::vector<float> h_a_out(h_a_ref.size(), 0.0F);
  std::vector<float> h_b_out(h_b_ref.size(), 0.0F);
  auto get_ptr = [&](const char* name) -> void* {
    for (int i = 0; i < nb; ++i) {
      if (std::string(io_names[i]) == name) {
        return io_ptrs[i];
      }
    }
    return nullptr;
  };
  cudaMemcpy(h_g_out.data(), get_ptr("out_gather"), h_g_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_a_out.data(), get_ptr("out_align"), h_a_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_b_out.data(), get_ptr("out_bevpool"), h_b_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < static_cast<int>(h_g_ref.size()); ++i) {
    if (!NearlyEqual(h_g_out[i], h_g_ref[i])) {
      PrintMismatch("GatherBev TRT", i, h_g_out[i], h_g_ref[i]);
      ok = false;
    }
  }
  for (int i = 0; i < static_cast<int>(h_a_ref.size()); ++i) {
    if (!NearlyEqual(h_a_out[i], h_a_ref[i])) {
      PrintMismatch("AlignBev TRT", i, h_a_out[i], h_a_ref[i]);
      ok = false;
    }
  }
  for (int i = 0; i < static_cast<int>(h_b_ref.size()); ++i) {
    if (!NearlyEqual(h_b_out[i], h_b_ref[i])) {
      PrintMismatch("BevPool TRT", i, h_b_out[i], h_b_ref[i]);
      ok = false;
    }
  }

  for (void* p : io_ptrs) {
    cudaFree(p);
  }
  cudaStreamDestroy(stream);
  if (!ok) {
    std::cerr << "Minimal TRT plugin integration test FAILED\n";
    return 1;
  }
  std::cout << "Minimal TRT plugin integration test PASSED\n";
  return 0;
}
