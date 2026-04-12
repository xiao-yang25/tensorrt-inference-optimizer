#include "alignbev_plugin.h"

#include <cuda_runtime_api.h>

#include <cstring>

namespace tio {

namespace {
constexpr char kPluginName[] = "AlignBevPlugin";
constexpr char kPluginVersion[] = "1";
constexpr int kInputCount = 2;
constexpr int kOutputCount = 1;
}  // namespace

AlignBevPlugin::AlignBevPlugin(const std::string& name) : layer_name_(name) {}

AlignBevPlugin::AlignBevPlugin(const void* serial_data, std::size_t serial_length) {
  layer_name_.assign(static_cast<const char*>(serial_data), serial_length);
}

int AlignBevPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs AlignBevPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int,
                                                        nvinfer1::IExprBuilder&) noexcept {
  return inputs[0];
}

bool AlignBevPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int,
                                               int) noexcept {
  const auto linear = nvinfer1::TensorFormat::kLINEAR;
  if (pos < 0 || pos >= kInputCount + kOutputCount) {
    return false;
  }
  if (pos == 1) {  // transforms
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == linear;
  }
  return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == linear;
}

void AlignBevPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                                     const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}

std::size_t AlignBevPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                             const nvinfer1::PluginTensorDesc*, int) const noexcept {
  return 0;
}

int AlignBevPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*,
                            const void* const* inputs, void* const* outputs, void*,
                            cudaStream_t stream) noexcept {
  if (inputDesc[0].dims.nbDims != 4) {
    return -1;
  }
  const int adj_num = inputDesc[0].dims.d[0];
  const int channels = inputDesc[0].dims.d[1];
  const int bev_h = inputDesc[0].dims.d[2];
  const int bev_w = inputDesc[0].dims.d[3];
  if (adj_num <= 0 || channels <= 0 || bev_h <= 0 || bev_w <= 0) {
    return -1;
  }

  LaunchAlignBevKernel(static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
                       static_cast<float*>(outputs[0]), adj_num, channels, bev_h, bev_w, stream);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return -1;
  }
  return 0;
}

std::size_t AlignBevPlugin::getSerializationSize() const noexcept { return layer_name_.size(); }

void AlignBevPlugin::serialize(void* buffer) const noexcept {
  std::memcpy(buffer, layer_name_.data(), layer_name_.size());
}

nvinfer1::IPluginV2DynamicExt* AlignBevPlugin::clone() const noexcept {
  auto* plugin = new AlignBevPlugin(layer_name_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void AlignBevPlugin::destroy() noexcept { delete this; }

int AlignBevPlugin::initialize() noexcept { return 0; }

void AlignBevPlugin::terminate() noexcept {}

void AlignBevPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* AlignBevPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

const char* AlignBevPlugin::getPluginType() const noexcept { return kPluginName; }

const char* AlignBevPlugin::getPluginVersion() const noexcept { return kPluginVersion; }

nvinfer1::DataType AlignBevPlugin::getOutputDataType(int, const nvinfer1::DataType* inputTypes,
                                                     int) const noexcept {
  return inputTypes[0];
}

AlignBevPluginCreator::AlignBevPluginCreator() {
  fields_.nbFields = 0;
  fields_.fields = attrs_.data();
}

const char* AlignBevPluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* AlignBevPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
const nvinfer1::PluginFieldCollection* AlignBevPluginCreator::getFieldNames() noexcept { return &fields_; }

nvinfer1::IPluginV2* AlignBevPluginCreator::createPlugin(const char* name,
                                                         const nvinfer1::PluginFieldCollection*) noexcept {
  auto* plugin = new AlignBevPlugin(name ? name : "alignbev_plugin");
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* AlignBevPluginCreator::deserializePlugin(const char*, const void* serialData,
                                                              std::size_t serialLength) noexcept {
  auto* plugin = new AlignBevPlugin(serialData, serialLength);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void AlignBevPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* AlignBevPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }

}  // namespace tio
