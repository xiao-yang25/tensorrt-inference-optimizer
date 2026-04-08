#pragma once

#include <NvInferRuntimeCommon.h>

#include <cstddef>
#include <string>
#include <vector>

namespace tio {

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator(std::size_t batch_bytes, std::vector<std::string> batch_files,
                        std::string cache_file);
  ~Int8EntropyCalibrator() override;

  int getBatchSize() const noexcept override;
  bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
  const void* readCalibrationCache(std::size_t& length) noexcept override;
  void writeCalibrationCache(const void* cache, std::size_t length) noexcept override;

 private:
  bool LoadBatch(const std::string& path, std::vector<char>* out);

  std::size_t batch_bytes_{0};
  std::vector<std::string> batch_files_;
  std::string cache_file_;
  std::size_t cursor_{0};
  void* device_batch_{nullptr};
  std::vector<char> cache_;
};

}  // namespace tio
