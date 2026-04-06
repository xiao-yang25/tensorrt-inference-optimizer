#include "config.h"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace tio {

namespace {

std::vector<int> ParseIntArray(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    return {};
  }
  return node[key].as<std::vector<int>>();
}

}  // namespace

AppConfig LoadConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  AppConfig cfg;

  if (root["build"]) {
    const auto build = root["build"];
    cfg.build.fp16 = build["fp16"] ? build["fp16"].as<bool>() : cfg.build.fp16;
    cfg.build.int8 = build["int8"] ? build["int8"].as<bool>() : cfg.build.int8;
    cfg.build.workspace_mb =
        build["workspace_mb"] ? build["workspace_mb"].as<std::size_t>() : cfg.build.workspace_mb;
    cfg.build.onnx_path = build["onnx_path"] ? build["onnx_path"].as<std::string>() : "";
    cfg.build.engine_path = build["engine_path"] ? build["engine_path"].as<std::string>() : "";
    cfg.build.calibrator_cache_path =
        build["calibrator_cache_path"] ? build["calibrator_cache_path"].as<std::string>() : "";
    cfg.build.input_tensor_name =
        build["input_tensor_name"] ? build["input_tensor_name"].as<std::string>()
                                   : cfg.build.input_tensor_name;

    if (build["calibration_batches"]) {
      cfg.build.calibration_batches = build["calibration_batches"].as<std::vector<std::string>>();
    }

    if (build["profile"]) {
      cfg.build.profile.min = ParseIntArray(build["profile"], "min");
      cfg.build.profile.opt = ParseIntArray(build["profile"], "opt");
      cfg.build.profile.max = ParseIntArray(build["profile"], "max");
    }
  }

  if (root["runtime"]) {
    const auto runtime = root["runtime"];
    cfg.runtime.warmup_runs =
        runtime["warmup_runs"] ? runtime["warmup_runs"].as<int>() : cfg.runtime.warmup_runs;
    cfg.runtime.benchmark_runs =
        runtime["benchmark_runs"] ? runtime["benchmark_runs"].as<int>() : cfg.runtime.benchmark_runs;
    cfg.runtime.batch = runtime["batch"] ? runtime["batch"].as<int>() : cfg.runtime.batch;
    cfg.runtime.use_async = runtime["use_async"] ? runtime["use_async"].as<bool>() : cfg.runtime.use_async;
    cfg.runtime.stream_count =
        runtime["stream_count"] ? runtime["stream_count"].as<int>() : cfg.runtime.stream_count;
  }

  if (cfg.build.engine_path.empty()) {
    throw std::runtime_error("Config missing required key: build.engine_path");
  }
  return cfg;
}

}  // namespace tio
