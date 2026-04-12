#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <string>
#include <vector>

namespace tio {

void LaunchGatherBevKernel(const float* adj_feat, const float* curr_feat, const int* flags, float* out_feat, int b,
                           int adj_num, int channels, int map_size, cudaStream_t stream);

class GatherBevPlugin final : public nvinfer1::IPluginV2DynamicExt {
 public:
  using nvinfer1::IPluginV2Ext::configurePlugin;
  using nvinfer1::IPluginV2::enqueue;
  using nvinfer1::IPluginV2::getOutputDimensions;
  using nvinfer1::IPluginV2::getWorkspaceSize;

  GatherBevPlugin() = default;
  explicit GatherBevPlugin(const std::string& name);
  GatherBevPlugin(const void* serial_data, std::size_t serial_length);
  ~GatherBevPlugin() override = default;

  int getNbOutputs() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
                                          nvinfer1::IExprBuilder& exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
  std::size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                               const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
              const void* const* inputs, void* const* outputs, void* workspace,
              cudaStream_t stream) noexcept override;

  std::size_t getSerializationSize() const noexcept override;
  void serialize(void* buffer) const noexcept override;
  nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

  void destroy() noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;
  const char* getPluginType() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes,
                                       int nbInputs) const noexcept override;
  void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {}
  void detachFromContext() noexcept override {}

 private:
  std::string layer_name_{"gatherbev_plugin"};
  std::string namespace_;
};

class GatherBevPluginCreator final : public nvinfer1::IPluginCreator {
 public:
  GatherBevPluginCreator();
  ~GatherBevPluginCreator() override = default;

  const char* getPluginName() const noexcept override;
  const char* getPluginVersion() const noexcept override;
  const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
  nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
  nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData,
                                         std::size_t serialLength) noexcept override;
  void setPluginNamespace(const char* pluginNamespace) noexcept override;
  const char* getPluginNamespace() const noexcept override;

 private:
  std::string namespace_;
  nvinfer1::PluginFieldCollection fields_{};
  std::vector<nvinfer1::PluginField> attrs_{};
};

}  // namespace tio
