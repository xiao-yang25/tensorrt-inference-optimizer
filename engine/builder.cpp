#include "builder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "calibrator.h"
#include "utils.h"

namespace tio {

namespace {

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, void (*)(T*)>;

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
TrtUniquePtr<T> MakeTrt(T* p) {
  return TrtUniquePtr<T>(p, [](T* obj) { DestroyTrtObject(obj); });
}

}  // namespace

EngineBuilder::EngineBuilder(TrtLogger* logger) : logger_(logger) {}
EngineBuilder::~EngineBuilder() = default;

bool EngineBuilder::BuildAndSerialize(const BuildConfig& cfg) {
  const auto plan = BuildSerializedPlan(cfg);
  return SaveBinaryFile(cfg.engine_path, plan);
}

std::vector<char> EngineBuilder::BuildSerializedPlan(const BuildConfig& cfg) {
  auto builder = MakeTrt(nvinfer1::createInferBuilder(*logger_));
  if (!builder) {
    throw std::runtime_error("Failed to create TensorRT builder.");
  }

  const auto network_flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = MakeTrt(builder->createNetworkV2(network_flags));
  auto parser = MakeTrt(nvonnxparser::createParser(*network, *logger_));

  if (cfg.onnx_path.empty()) {
    throw std::runtime_error("build.onnx_path is required when building engine.");
  }
  if (!parser->parseFromFile(cfg.onnx_path.c_str(),
                             static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("Failed to parse ONNX model: " + cfg.onnx_path);
  }

  auto config = MakeTrt(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, cfg.workspace_mb * 1024ULL * 1024ULL);

  if (cfg.fp16 && SupportsFp16(builder.get())) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (cfg.int8) {
    if (!SupportsInt8(builder.get())) {
      throw std::runtime_error("Requested INT8 but device does not support INT8.");
    }
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    if (network->getNbInputs() < 1) {
      throw std::runtime_error("Network has no input for INT8 calibration.");
    }
    auto input = network->getInput(0);
    const auto batch_bytes = static_cast<std::size_t>(Volume(input->getDimensions())) * ElementSize(input->getType());
    calibrator_ = std::make_unique<Int8EntropyCalibrator>(
        batch_bytes, cfg.calibration_batches, cfg.calibrator_cache_path);
    config->setInt8Calibrator(calibrator_.get());
  }

  if (!cfg.profile.opt.empty()) {
    auto profile = builder->createOptimizationProfile();
    auto dims_min = DimsFromVector(cfg.profile.min);
    auto dims_opt = DimsFromVector(cfg.profile.opt);
    auto dims_max = DimsFromVector(cfg.profile.max);

    if (!profile->setDimensions(cfg.input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min) ||
        !profile->setDimensions(cfg.input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt) ||
        !profile->setDimensions(cfg.input_tensor_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max)) {
#if NV_TENSORRT_MAJOR < 10
      profile->destroy();
#endif
      throw std::runtime_error("Failed to set optimization profile dimensions.");
    }
    config->addOptimizationProfile(profile);
#if NV_TENSORRT_MAJOR < 10
    profile->destroy();
#endif
  }

  auto plan = MakeTrt(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    throw std::runtime_error("Failed to build serialized TensorRT engine.");
  }

  const auto* ptr = static_cast<const char*>(plan->data());
  return std::vector<char>(ptr, ptr + plan->size());
}

nvinfer1::Dims EngineBuilder::DimsFromVector(const std::vector<int>& shape) {
  nvinfer1::Dims dims{};
  dims.nbDims = static_cast<int>(shape.size());
  for (int i = 0; i < dims.nbDims; ++i) {
    dims.d[i] = shape.at(i);
  }
  return dims;
}

bool EngineBuilder::SupportsFp16(nvinfer1::IBuilder* builder) const {
  return builder->platformHasFastFp16();
}

bool EngineBuilder::SupportsInt8(nvinfer1::IBuilder* builder) const {
  return builder->platformHasFastInt8();
}

bool SaveBinaryFile(const std::string& path, const std::vector<char>& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }
  ofs.write(data.data(), static_cast<std::streamsize>(data.size()));
  return ofs.good();
}

std::vector<char> ReadBinaryFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Unable to read file: " + path);
  }
  return std::vector<char>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

}  // namespace tio
