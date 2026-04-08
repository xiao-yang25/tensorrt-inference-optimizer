#include "calibrator.h"

#include <cuda_runtime_api.h>

#include <cstring>
#include <fstream>
#include <iostream>

namespace tio {

Int8EntropyCalibrator::Int8EntropyCalibrator(std::size_t batch_bytes,
                                             std::vector<std::string> batch_files,
                                             std::string cache_file)
    : batch_bytes_(batch_bytes),
      batch_files_(std::move(batch_files)),
      cache_file_(std::move(cache_file)) {
  cudaMalloc(&device_batch_, batch_bytes_);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
  if (device_batch_) {
    cudaFree(device_batch_);
  }
}

int Int8EntropyCalibrator::getBatchSize() const noexcept { return 1; }

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char*[], int) noexcept {
  if (cursor_ >= batch_files_.size()) {
    return false;
  }

  std::vector<char> host_batch;
  if (!LoadBatch(batch_files_[cursor_], &host_batch)) {
    return false;
  }
  ++cursor_;

  if (host_batch.size() != batch_bytes_) {
    std::cerr << "Calibration batch size mismatch. Expected " << batch_bytes_ << ", got "
              << host_batch.size() << std::endl;
    return false;
  }

  if (cudaMemcpy(device_batch_, host_batch.data(), batch_bytes_, cudaMemcpyHostToDevice) != cudaSuccess) {
    return false;
  }

  bindings[0] = device_batch_;
  return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(std::size_t& length) noexcept {
  cache_.clear();
  if (cache_file_.empty()) {
    length = 0;
    return nullptr;
  }
  std::ifstream ifs(cache_file_, std::ios::binary);
  if (!ifs.good()) {
    length = 0;
    return nullptr;
  }
  cache_ = std::vector<char>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  length = cache_.size();
  return cache_.empty() ? nullptr : cache_.data();
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, std::size_t length) noexcept {
  if (cache_file_.empty() || cache == nullptr || length == 0) {
    return;
  }
  std::ofstream ofs(cache_file_, std::ios::binary);
  if (!ofs.good()) {
    return;
  }
  ofs.write(reinterpret_cast<const char*>(cache), static_cast<std::streamsize>(length));
}

bool Int8EntropyCalibrator::LoadBatch(const std::string& path, std::vector<char>* out) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.good()) {
    std::cerr << "Unable to read calibration batch: " << path << std::endl;
    return false;
  }
  *out = std::vector<char>((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  return true;
}

}  // namespace tio
