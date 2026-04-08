#include "bevpool_plugin.h"

#include <cuda_runtime_api.h>

#include <cstring>

namespace tio {

namespace {
constexpr char kPluginName[] = "BevPoolPlugin";
constexpr char kPluginVersion[] = "1";
}  // namespace

BevPoolPlugin::BevPoolPlugin(const std::string& name) : layer_name_(name) {}

BevPoolPlugin::BevPoolPlugin(const void* serial_data, std::size_t serial_length) {
  layer_name_.assign(static_cast<const char*>(serial_data), serial_length);
}

int BevPoolPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs BevPoolPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int,
                                                       nvinfer1::IExprBuilder&) noexcept {
  return inputs[0];
}

bool BevPoolPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int, int) noexcept {
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void BevPoolPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                                    const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}

std::size_t BevPoolPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                            const nvinfer1::PluginTensorDesc*, int) const noexcept {
  return 0;
}

int BevPoolPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*,
                           const void* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept {
  int element_count = 1;
  for (int i = 0; i < inputDesc[0].dims.nbDims; ++i) {
    element_count *= inputDesc[0].dims.d[i];
  }
  LaunchBevPoolIdentityKernel(static_cast<const float*>(inputs[0]), static_cast<float*>(outputs[0]), element_count,
                              stream);
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
