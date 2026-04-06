#pragma once

#include <string>
#include <vector>

namespace tio {

struct ProfileShape {
  std::vector<int> min;
  std::vector<int> opt;
  std::vector<int> max;
};

struct BuildConfig {
  bool fp16{false};
  bool int8{false};
  std::size_t workspace_mb{2048};
  std::string onnx_path;
  std::string engine_path;
  std::string calibrator_cache_path;
  std::vector<std::string> calibration_batches;
  std::string input_tensor_name{"input"};
  ProfileShape profile{};
};

struct RuntimeConfig {
  int warmup_runs{20};
  int benchmark_runs{200};
  int batch{1};
  bool use_async{true};
  int stream_count{1};
};

struct AppConfig {
  BuildConfig build;
  RuntimeConfig runtime;
};

AppConfig LoadConfig(const std::string& path);

}  // namespace tio
