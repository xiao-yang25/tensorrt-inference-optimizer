#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "buffer_manager.h"
#include "logger.h"

namespace tio {

class InferRunner {
 public:
  explicit InferRunner(TrtLogger* logger);
  ~InferRunner();

  bool LoadEngineFromFile(const std::string& engine_path);
  bool PrepareBindings(int batch_size);
  bool RunOnce(cudaStream_t stream);

  nvinfer1::ICudaEngine* Engine() const { return engine_.get(); }
  nvinfer1::IExecutionContext* Context() const { return context_.get(); }
  BufferManager& Buffers() { return buffers_; }

 private:
  template <typename T>
  using TrtUniquePtr = std::unique_ptr<T, void (*)(T*)>;

  TrtLogger* logger_{nullptr};
  TrtUniquePtr<nvinfer1::IRuntime> runtime_{nullptr, nullptr};
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_{nullptr, nullptr};
  TrtUniquePtr<nvinfer1::IExecutionContext> context_{nullptr, nullptr};
  BufferManager buffers_;
};

}  // namespace tio
