#include "gatherbev_plugin.h"

#include <cuda_runtime_api.h>

#include <cstring>

namespace tio {

namespace {
constexpr char kPluginName[] = "GatherBevPlugin";
constexpr char kPluginVersion[] = "1";
constexpr int kInputCount = 3;
constexpr int kOutputCount = 1;
}  // namespace

GatherBevPlugin::GatherBevPlugin(const std::string& name) : layer_name_(name) {}

GatherBevPlugin::GatherBevPlugin(const void* serial_data, std::size_t serial_length) {
  layer_name_.assign(static_cast<const char*>(serial_data), serial_length);
}

int GatherBevPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::DimsExprs GatherBevPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int,
                                                         nvinfer1::IExprBuilder& exprBuilder) noexcept {
  // in0: [B,Adj,C,H,W] -> out: [B,(Adj+1)*C,H,W]
  nvinfer1::DimsExprs ret{};
  ret.nbDims = 4;
  auto* adj_plus_1 = exprBuilder.operation(nvinfer1::DimensionOperation::kSUM, *inputs[0].d[1],
                                           *exprBuilder.constant(1));
  ret.d[0] = inputs[0].d[0];
  ret.d[1] = exprBuilder.operation(nvinfer1::DimensionOperation::kPROD, *adj_plus_1, *inputs[0].d[2]);
  ret.d[2] = inputs[0].d[3];
  ret.d[3] = inputs[0].d[4];
  return ret;
}

bool GatherBevPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int,
                                                int) noexcept {
  const auto linear = nvinfer1::TensorFormat::kLINEAR;
  if (pos < 0 || pos >= kInputCount + kOutputCount) {
    return false;
  }
  if (pos == 2) {  // flags
    return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == linear;
  }
  if (pos == 0 || pos == 1 || pos == 3) {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT && inOut[pos].format == linear;
  }
  return false;
}

void GatherBevPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int,
                                      const nvinfer1::DynamicPluginTensorDesc*, int) noexcept {}

std::size_t GatherBevPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int,
                                              const nvinfer1::PluginTensorDesc*, int) const noexcept {
  return 0;
}

int GatherBevPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc*,
                             const void* const* inputs, void* const* outputs, void*,
                             cudaStream_t stream) noexcept {
  if (inputDesc[0].dims.nbDims != 5 || inputDesc[1].dims.nbDims != 4) {
    return -1;
  }
  const int b = inputDesc[0].dims.d[0];
  const int adj_num = inputDesc[0].dims.d[1];
  const int channels = inputDesc[0].dims.d[2];
  const int map_size = inputDesc[0].dims.d[3] * inputDesc[0].dims.d[4];
  if (b <= 0 || adj_num <= 0 || channels <= 0 || map_size <= 0) {
    return -1;
  }
  LaunchGatherBevKernel(static_cast<const float*>(inputs[0]), static_cast<const float*>(inputs[1]),
                        static_cast<const int*>(inputs[2]), static_cast<float*>(outputs[0]), b, adj_num, channels,
                        map_size, stream);
  if (cudaPeekAtLastError() != cudaSuccess) {
    return -1;
  }
  return 0;
}

std::size_t GatherBevPlugin::getSerializationSize() const noexcept { return layer_name_.size(); }

void GatherBevPlugin::serialize(void* buffer) const noexcept {
  std::memcpy(buffer, layer_name_.data(), layer_name_.size());
}

nvinfer1::IPluginV2DynamicExt* GatherBevPlugin::clone() const noexcept {
  auto* plugin = new GatherBevPlugin(layer_name_);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void GatherBevPlugin::destroy() noexcept { delete this; }

int GatherBevPlugin::initialize() noexcept { return 0; }

void GatherBevPlugin::terminate() noexcept {}

void GatherBevPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GatherBevPlugin::getPluginNamespace() const noexcept { return namespace_.c_str(); }

const char* GatherBevPlugin::getPluginType() const noexcept { return kPluginName; }

const char* GatherBevPlugin::getPluginVersion() const noexcept { return kPluginVersion; }

nvinfer1::DataType GatherBevPlugin::getOutputDataType(int, const nvinfer1::DataType* inputTypes,
                                                      int) const noexcept {
  return inputTypes[0];
}

GatherBevPluginCreator::GatherBevPluginCreator() {
  fields_.nbFields = 0;
  fields_.fields = attrs_.data();
}

const char* GatherBevPluginCreator::getPluginName() const noexcept { return kPluginName; }
const char* GatherBevPluginCreator::getPluginVersion() const noexcept { return kPluginVersion; }
const nvinfer1::PluginFieldCollection* GatherBevPluginCreator::getFieldNames() noexcept { return &fields_; }

nvinfer1::IPluginV2* GatherBevPluginCreator::createPlugin(const char* name,
                                                          const nvinfer1::PluginFieldCollection*) noexcept {
  auto* plugin = new GatherBevPlugin(name ? name : "gatherbev_plugin");
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

nvinfer1::IPluginV2* GatherBevPluginCreator::deserializePlugin(const char*, const void* serialData,
                                                               std::size_t serialLength) noexcept {
  auto* plugin = new GatherBevPlugin(serialData, serialLength);
  plugin->setPluginNamespace(namespace_.c_str());
  return plugin;
}

void GatherBevPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
  namespace_ = pluginNamespace ? pluginNamespace : "";
}

const char* GatherBevPluginCreator::getPluginNamespace() const noexcept { return namespace_.c_str(); }

}  // namespace tio
