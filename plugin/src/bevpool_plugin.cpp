#include "bevpool_plugin.h"

#include <cuda_runtime_api.h>

#include <cstring>

namespace tio {

namespace {
constexpr char kPluginName[] = "BevPoolPlugin";
constexpr char kPluginVersion[] = "1";
constexpr int kInputCount = 8;
constexpr int kOutputCount = 1;
}  // namespace

BevPoolPlugin::BevPoolPlugin(const std::string& name) : layer_name_(name) {}

BevPoolPlugin::BevPoolPlugin(const void* serial_data, std::size_t serial_length) {
  layer_name_.assign(static_cast<const char*>(serial_data), serial_length);
}

int BevPoolPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs BevPoolPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int,
                                                       nvinfer1::IExprBuilder&) noexcept {
  // Input-0 is the BEV template tensor with target output shape.
  return inputs[0];
}

bool BevPoolPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int, int) noexcept {
  const auto linear = nvinfer1::TensorFormat::kLINEAR;
  if (pos < 0 || pos >= kInputCount + kOutputCount) {
    return false;
  }
  if (pos <= 2 || pos == 8) {  // bev_template, depth, feat, output
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == linear;
  }
  return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == linear;
}

void BevPoolPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                                    const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}

std::size_t BevPoolPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                            const nvinfer1::PluginTensorDesc*, int) const noexcept {
  return 0;
}

int BevPoolPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*,
                           const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
  // Inputs:
  // 0: bev template (float, [N,C,H,W]), only shape used
  // 1: depth
  // 2: feat
  // 3..7: ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths (int32)
  int map_size = 1;
  if (inputDesc[0].dims.nbDims < 4) {
    return -1;
  }
  const int channels = inputDesc[0].dims.d[1];
  map_size = inputDesc[0].dims.d[2] * inputDesc[0].dims.d[3];

  int n_intervals = 1;
  for (int i = 0; i < inputDesc[7].dims.nbDims; ++i) {
    n_intervals *= inputDesc[7].dims.d[i];
  }
  if (channels <= 0 || map_size <= 0 || n_intervals <= 0) {
    return -1;
  }

  LaunchBevPoolKernel(
      channels, n_intervals, map_size, static_cast<const float*>(inputs[1]), static_cast<const float*>(inputs[2]),
      static_cast<const int*>(inputs[3]), static_cast<const int*>(inputs[4]), static_cast<const int*>(inputs[5]),
      static_cast<const int*>(inputs[6]), static_cast<const int*>(inputs[7]), static_cast<float*>(outputs[0]),
      stream);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return -1;
  }
  return 0;
}

std::size_t BevPoolPlugin::getSerializationSize() const noexcept { return layer_name_.size(); }

void BevPoolPlugin::serialize(void* buffer) const noexcept {
  std::memcpy(buffer, layer_name_.data(), layer_name_.size());
}

nvinfer1::IPluginV2DynamicExt* BevPoolPlugin::clone() const noexcept {
  auto* plugin = new BevPoolPlugin(layer_name_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void BevPoolPlugin::destroy() noexcept { delete this; }

int BevPoolPlugin::initialize() noexcept { return 0; }

void BevPoolPlugin::terminate() noexcept {}

void BevPoolPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* BevPoolPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

const char* BevPoolPlugin::getPluginType() const noexcept { return kPluginName; }

const char* BevPoolPlugin::getPluginVersion() const noexcept { return kPluginVersion; }

nvinfer1::DataType BevPoolPlugin::getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept {
  return inputTypes[0];
}

BevPoolPluginCreator::BevPoolPluginCreator() {
  fields_.nbFields = 0;
  fields_.fields = attrs_.data();
}

const char* BevPoolPluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* BevPoolPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
const nvinfer1::PluginFieldCollection* BevPoolPluginCreator::getFieldNames() noexcept { return &fields_; }

nvinfer1::IPluginV2* BevPoolPluginCreator::createPlugin(const char* name,
                                                        const nvinfer1::PluginFieldCollection*) noexcept {
  auto* plugin = new BevPoolPlugin(name ? name : "bevpool_plugin");
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* BevPoolPluginCreator::deserializePlugin(const char*,
                                                             const void* serialData,
                                                             std::size_t serialLength) noexcept {
  auto* plugin = new BevPoolPlugin(serialData, serialLength);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void BevPoolPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* BevPoolPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }

}  // namespace tio
