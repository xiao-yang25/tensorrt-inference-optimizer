#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "buffer_manager.h"
#include "infer.h"
#include "logger.h"

namespace {

struct ErrorStats {
  double mae{0.0};
  double rmse{0.0};
  double max_abs{0.0};
  std::size_t count{0};
  std::size_t mismatched{0};
};

float HalfToFloat(std::uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(h));
  return __half2float(h);
}

std::uint16_t FloatToHalf(float x) {
  const __half h = __float2half(x);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

bool FillAndCopyInput(const tio::DeviceBinding& a, const tio::DeviceBinding& b, std::mt19937& rng) {
  if (a.type != b.type || a.bytes != b.bytes) {
    std::cerr << "Input tensor mismatch on `" << a.name << "`\n";
    return false;
  }
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  if (a.type == nvinfer1::DataType::kFLOAT) {
    std::vector<float> host(a.bytes / sizeof(float));
    for (float& v : host) {
      v = dist(rng);
    }
    return cudaMemcpy(a.ptr, host.data(), a.bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(b.ptr, host.data(), b.bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  if (a.type == nvinfer1::DataType::kHALF) {
    std::vector<std::uint16_t> host(a.bytes / sizeof(std::uint16_t));
    for (auto& bits : host) {
      bits = FloatToHalf(dist(rng));
    }
    return cudaMemcpy(a.ptr, host.data(), a.bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(b.ptr, host.data(), b.bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  if (a.type == nvinfer1::DataType::kINT32) {
    std::vector<int> host(a.bytes / sizeof(int));
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i] = static_cast<int>(i % 97);
    }
    return cudaMemcpy(a.ptr, host.data(), a.bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(b.ptr, host.data(), b.bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  std::cerr << "Unsupported input dtype on `" << a.name << "`\n";
  return false;
}

ErrorStats CompareOutput(const tio::DeviceBinding& a, const tio::DeviceBinding& b) {
  ErrorStats stats{};
  if (a.type != b.type || a.bytes != b.bytes) {
    stats.mismatched = 1;
    return stats;
  }
  if (a.type == nvinfer1::DataType::kFLOAT) {
    std::vector<float> va(a.bytes / sizeof(float));
    std::vector<float> vb(b.bytes / sizeof(float));
    cudaMemcpy(va.data(), a.ptr, a.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), b.ptr, b.bytes, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < va.size(); ++i) {
      const double diff = std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
      stats.mae += diff;
      stats.rmse += diff * diff;
      stats.max_abs = std::max(stats.max_abs, diff);
      if (diff > 1e-5) {
        ++stats.mismatched;
      }
    }
    stats.count = va.size();
    return stats;
  }
  if (a.type == nvinfer1::DataType::kHALF) {
    std::vector<std::uint16_t> va(a.bytes / sizeof(std::uint16_t));
    std::vector<std::uint16_t> vb(b.bytes / sizeof(std::uint16_t));
    cudaMemcpy(va.data(), a.ptr, a.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), b.ptr, b.bytes, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < va.size(); ++i) {
      const double diff =
          std::abs(static_cast<double>(HalfToFloat(va[i])) - static_cast<double>(HalfToFloat(vb[i])));
      stats.mae += diff;
      stats.rmse += diff * diff;
      stats.max_abs = std::max(stats.max_abs, diff);
      if (diff > 1e-3) {
        ++stats.mismatched;
      }
    }
    stats.count = va.size();
    return stats;
  }
  if (a.type == nvinfer1::DataType::kINT32) {
    std::vector<int> va(a.bytes / sizeof(int));
    std::vector<int> vb(b.bytes / sizeof(int));
    cudaMemcpy(va.data(), a.ptr, a.bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vb.data(), b.ptr, b.bytes, cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < va.size(); ++i) {
      const double diff = std::abs(static_cast<double>(va[i]) - static_cast<double>(vb[i]));
      stats.mae += diff;
      stats.rmse += diff * diff;
      stats.max_abs = std::max(stats.max_abs, diff);
      if (va[i] != vb[i]) {
        ++stats.mismatched;
      }
    }
    stats.count = va.size();
    return stats;
  }
  stats.mismatched = 1;
  return stats;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: tio_compare_engines <engine_a> <engine_b>\n";
    return 1;
  }
  tio::TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
  tio::InferRunner runner_a(&logger);
  tio::InferRunner runner_b(&logger);
  if (!runner_a.LoadEngineFromFile(argv[1]) || !runner_b.LoadEngineFromFile(argv[2])) {
    std::cerr << "Failed to load engines.\n";
    return 2;
  }
  if (!runner_a.PrepareBindings(1) || !runner_b.PrepareBindings(1)) {
    std::cerr << "Failed to allocate engine bindings.\n";
    return 3;
  }

  std::mt19937 rng(20260411U);
  for (const auto& in_a : runner_a.Buffers().DeviceBindings()) {
    if (!in_a.is_input) {
      continue;
    }
    const auto* in_b = runner_b.Buffers().GetBinding(in_a.name);
    if (!in_b || !in_b->is_input) {
      std::cerr << "Input tensor `" << in_a.name << "` not found in engine_b.\n";
      return 4;
    }
    if (!FillAndCopyInput(in_a, *in_b, rng)) {
      std::cerr << "Failed to prepare input `" << in_a.name << "`.\n";
      return 5;
    }
  }

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  if (!runner_a.RunOnce(stream) || !runner_b.RunOnce(stream)) {
    std::cerr << "Engine run failed.\n";
    cudaStreamDestroy(stream);
    return 6;
  }
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  double sum_mae = 0.0;
  double sum_rmse = 0.0;
  double max_abs = 0.0;
  std::size_t sum_count = 0;
  std::size_t sum_mismatch = 0;

  std::cout << std::fixed << std::setprecision(6);
  for (const auto& out_a : runner_a.Buffers().DeviceBindings()) {
    if (out_a.is_input) {
      continue;
    }
    const auto* out_b = runner_b.Buffers().GetBinding(out_a.name);
    if (!out_b || out_b->is_input) {
      std::cerr << "Output tensor `" << out_a.name << "` not found in engine_b.\n";
      return 7;
    }
    const ErrorStats s = CompareOutput(out_a, *out_b);
    if (s.count == 0 && s.mismatched > 0) {
      std::cerr << "Unsupported output dtype on `" << out_a.name << "`.\n";
      return 8;
    }
    const double mae = s.count ? s.mae / static_cast<double>(s.count) : 0.0;
    const double rmse = s.count ? std::sqrt(s.rmse / static_cast<double>(s.count)) : 0.0;
    std::cout << "output=" << out_a.name << " mae=" << mae << " rmse=" << rmse << " max_abs=" << s.max_abs
              << " mismatched=" << s.mismatched << "/" << s.count << "\n";
    sum_mae += s.mae;
    sum_rmse += s.rmse;
    max_abs = std::max(max_abs, s.max_abs);
    sum_count += s.count;
    sum_mismatch += s.mismatched;
  }

  if (sum_count == 0) {
    std::cerr << "No comparable outputs found.\n";
    return 9;
  }
  const double overall_mae = sum_mae / static_cast<double>(sum_count);
  const double overall_rmse = std::sqrt(sum_rmse / static_cast<double>(sum_count));
  std::cout << "overall_mae=" << overall_mae << "\n";
  std::cout << "overall_rmse=" << overall_rmse << "\n";
  std::cout << "overall_max_abs=" << max_abs << "\n";
  std::cout << "overall_mismatched=" << sum_mismatch << "/" << sum_count << "\n";
  return 0;
}
