#include "infer.h"

#include <NvInfer.h>

#include <stdexcept>

#include "builder.h"

namespace tio {

namespace {
template <typename T>
void DestroyTrtObject(T* obj) noexcept {
  if (!obj) {
    return;
  }
#if NV_TENSORRT_MAJOR >= 10
  delete obj;
#else
  obj->destroy();
#endif
}

template <typename T>
std::unique_ptr<T, void (*)(T*)> MakeTrt(T* p) {
  return std::unique_ptr<T, void (*)(T*)>(p, [](T* obj) { DestroyTrtObject(obj); });
}
}  // namespace

InferRunner::InferRunner(TrtLogger* logger) : logger_(logger) {}

InferRunner::~InferRunner() = default;

bool InferRunner::LoadEngineFromFile(const std::string& engine_path) {
  const auto data = ReadBinaryFile(engine_path);

  runtime_ = MakeTrt(nvinfer1::createInferRuntime(*logger_));
  if (!runtime_) {
    return false;
  }
  engine_ = MakeTrt(runtime_->deserializeCudaEngine(data.data(), data.size()));
  if (!engine_) {
    return false;
  }
  context_ = MakeTrt(engine_->createExecutionContext());
  return context_ != nullptr;
}

bool InferRunner::PrepareBindings(int batch_size) {
  if (!engine_ || !context_) {
    return false;
  }
  return buffers_.Allocate(engine_.get(), context_.get(), batch_size);
}

bool InferRunner::RunOnce(cudaStream_t stream) {
  if (!context_) {
    return false;
  }
  const int nb = engine_->getNbIOTensors();
  auto& raw = buffers_.Bindings();
  for (int i = 0; i < nb; ++i) {
    const char* name = engine_->getIOTensorName(i);
    if (!context_->setTensorAddress(name, raw[i])) {
      return false;
    }
  }
#if NV_TENSORRT_MAJOR >= 8
  return context_->enqueueV3(stream);
#else
  return context_->enqueueV2(raw.data(), stream, nullptr);
#endif
}

}  // namespace tio
