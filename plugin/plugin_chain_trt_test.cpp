#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "alignbev_plugin.h"
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

}  // namespace

int main() {
  tio::TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);

  std::unique_ptr<nvinfer1::IPluginV2> gather_plugin(new tio::GatherBevPlugin("gather"));
  std::unique_ptr<nvinfer1::IPluginV2> align_plugin(new tio::AlignBevPlugin("align"));

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

  auto* t_g_adj = network->addInput("g_adj", nvinfer1::DataType::kFLOAT, MakeDims({1, 2, 1, 2, 2}));
  auto* t_g_curr = network->addInput("g_curr", nvinfer1::DataType::kFLOAT, MakeDims({1, 1, 2, 2}));
  auto* t_g_flag = network->addInput("g_flag", nvinfer1::DataType::kINT32, MakeDims({1}));
  auto* t_tf = network->addInput("transform", nvinfer1::DataType::kFLOAT, MakeDims({2, 9}));
  if (!t_g_adj || !t_g_curr || !t_g_flag || !t_tf) {
    std::cerr << "Input creation failed\n";
    return 1;
  }

  nvinfer1::ITensor* gather_inputs[] = {t_g_adj, t_g_curr, t_g_flag};
  auto* gather_layer = network->addPluginV2(gather_inputs, 3, *gather_plugin);
  if (!gather_layer) {
    std::cerr << "add Gather plugin failed\n";
    return 1;
  }

  // Gather output: [1, 3, 2, 2] where channel-0 is curr, channel-1..2 are adj queue.
  auto* slice_adj = network->addSlice(*gather_layer->getOutput(0), MakeDims({0, 1, 0, 0}), MakeDims({1, 2, 2, 2}),
                                      MakeDims({1, 1, 1, 1}));
  if (!slice_adj) {
    std::cerr << "addSlice failed\n";
    return 1;
  }
  auto* to_align = network->addShuffle(*slice_adj->getOutput(0));
  if (!to_align) {
    std::cerr << "addShuffle failed\n";
    return 1;
  }
  to_align->setReshapeDimensions(MakeDims({2, 1, 2, 2}));

  nvinfer1::ITensor* align_inputs[] = {to_align->getOutput(0), t_tf};
  auto* align_layer = network->addPluginV2(align_inputs, 2, *align_plugin);
  if (!align_layer) {
    std::cerr << "add Align plugin failed\n";
    return 1;
  }
  align_layer->getOutput(0)->setName("out_chain");
  network->markOutput(*align_layer->getOutput(0));

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
      std::cerr << "Unsupported tensor type: " << name << "\n";
      return 1;
    }
    cudaMalloc(&io_ptrs[i], bytes);
    cudaMemset(io_ptrs[i], 0, bytes);
    if (!context->setTensorAddress(name, io_ptrs[i])) {
      std::cerr << "setTensorAddress failed for " << name << "\n";
      return 1;
    }
  }

  std::vector<float> h_adj = {10, 11, 12, 13, 20, 21, 22, 23};
  std::vector<float> h_curr = {1, 2, 3, 4};
  std::vector<int> h_flag = {1};
  std::vector<float> h_tf(2 * 9, 0.0F);
  for (int i = 0; i < 2; ++i) {
    h_tf[i * 9 + 0] = 1.0F;
    h_tf[i * 9 + 4] = 1.0F;
    h_tf[i * 9 + 8] = 1.0F;
  }
  std::vector<float> h_ref = h_adj;  // identity align on gathered adj queue.

  auto copy_input = [&](const char* name, const void* src, std::size_t bytes) {
    int idx = -1;
    for (int i = 0; i < nb; ++i) {
      if (std::string(io_names[i]) == name) {
        idx = i;
        break;
      }
    }
    if (idx < 0) {
      return false;
    }
    return cudaMemcpy(io_ptrs[idx], src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  };

  if (!copy_input("g_adj", h_adj.data(), h_adj.size() * sizeof(float)) ||
      !copy_input("g_curr", h_curr.data(), h_curr.size() * sizeof(float)) ||
      !copy_input("g_flag", h_flag.data(), h_flag.size() * sizeof(int)) ||
      !copy_input("transform", h_tf.data(), h_tf.size() * sizeof(float))) {
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

  void* out_ptr = nullptr;
  for (int i = 0; i < nb; ++i) {
    if (std::string(io_names[i]) == "out_chain") {
      out_ptr = io_ptrs[i];
      break;
    }
  }
  if (!out_ptr) {
    std::cerr << "Output tensor not found\n";
    return 1;
  }

  std::vector<float> h_out(h_ref.size(), 0.0F);
  cudaMemcpy(h_out.data(), out_ptr, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  bool ok = true;
  for (int i = 0; i < static_cast<int>(h_ref.size()); ++i) {
    if (!NearlyEqual(h_out[i], h_ref[i])) {
      std::cerr << "[Gather->Align TRT] mismatch at " << i << ": got " << h_out[i] << ", expect " << h_ref[i]
                << "\n";
      ok = false;
    }
  }

  for (void* p : io_ptrs) {
    cudaFree(p);
  }
  cudaStreamDestroy(stream);
  if (!ok) {
    std::cerr << "Plugin chain TRT test FAILED\n";
    return 1;
  }
  std::cout << "Plugin chain TRT test PASSED\n";
  return 0;
}
