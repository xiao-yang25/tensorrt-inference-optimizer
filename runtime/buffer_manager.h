#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <vector>

namespace tio {

struct DeviceBinding {
  void* ptr{nullptr};
  std::size_t bytes{0};
  bool is_input{false};
};

class BufferManager {
 public:
  BufferManager() = default;
  ~BufferManager();

  bool Allocate(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context, int batch_size);
  void Release();

  std::vector<void*>& Bindings() { return raw_bindings_; }
  const std::vector<DeviceBinding>& DeviceBindings() const { return bindings_; }

 private:
  std::vector<DeviceBinding> bindings_;
  std::vector<void*> raw_bindings_;
};

}  // namespace tio
