#pragma once

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

#include "config.h"
#include "logger.h"

namespace tio {

class Int8EntropyCalibrator;

class EngineBuilder {
 public:
  explicit EngineBuilder(TrtLogger* logger);

  bool BuildAndSerialize(const BuildConfig& cfg);
  std::vector<char> BuildSerializedPlan(const BuildConfig& cfg);

 private:
  nvinfer1::Dims DimsFromVector(const std::vector<int>& shape);
  bool SupportsFp16(nvinfer1::IBuilder* builder) const;
  bool SupportsInt8(nvinfer1::IBuilder* builder) const;

  TrtLogger* logger_{nullptr};
  std::unique_ptr<Int8EntropyCalibrator> calibrator_;
};

bool SaveBinaryFile(const std::string& path, const std::vector<char>& data);
std::vector<char> ReadBinaryFile(const std::string& path);

}  // namespace tio
