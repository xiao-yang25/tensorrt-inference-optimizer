#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

namespace tio {

struct DeviceBinding {
  std::string name;
  void* ptr{nullptr};
  std::size_t bytes{0};
  bool is_input{false};
  nvinfer1::Dims dims{};
  nvinfer1::DataType type{nvinfer1::DataType::kFLOAT};
};

class BufferManager {
 public:
  BufferManager() = default;
  ~BufferManager();

  bool Allocate(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, int batch_size);
  void Release();

  std::vector<void*>& Bindings() { return raw_bindings_; }
  const std::vector<DeviceBinding>& DeviceBindings() const { return bindings_; }
  const DeviceBinding* GetBinding(const std::string& name) const;
  DeviceBinding* GetBinding(const std::string& name);

 private:
  std::vector<DeviceBinding> bindings_;
  std::vector<void*> raw_bindings_;
};

}  // namespace tio
