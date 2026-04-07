#include "buffer_manager.h"

#include <stdexcept>

#include "utils.h"

namespace tio {

BufferManager::~BufferManager() { Release(); }

bool BufferManager::Allocate(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* context,
                             int batch_size) {
  Release();
  const int nb = engine->getNbIOTensors();
  bindings_.resize(nb);
  raw_bindings_.resize(nb);

  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    auto dims = context->getTensorShape(name);
    if (dims.nbDims > 0 && dims.d[0] == -1) {
      dims.d[0] = batch_size;
      if (!context->setInputShape(name, dims)) {
        throw std::runtime_error("Failed to set dynamic input shape.");
      }
    }
  }

  for (int i = 0; i < nb; ++i) {
    const char* name = engine->getIOTensorName(i);
    const auto mode = engine->getTensorIOMode(name);
    const auto type = engine->getTensorDataType(name);
    auto dims = context->getTensorShape(name);
    const auto bytes = static_cast<std::size_t>(Volume(dims)) * ElementSize(type);

    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
      Release();
      return false;
    }
    bindings_[i] = DeviceBinding{
        ptr,
        bytes,
        mode == nvinfer1::TensorIOMode::kINPUT,
    };
    raw_bindings_[i] = ptr;
  }
  return true;
}

void BufferManager::Release() {
  for (auto& b : bindings_) {
    if (b.ptr) {
      cudaFree(b.ptr);
    }
  }
  bindings_.clear();
  raw_bindings_.clear();
}

}  // namespace tio
