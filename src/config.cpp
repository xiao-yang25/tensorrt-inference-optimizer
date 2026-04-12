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

PipelineMode ParsePipelineMode(const YAML::Node& root) {
  if (!root["pipeline_mode"]) {
    return PipelineMode::kOneEngine;
  }
  const auto value = root["pipeline_mode"].as<std::string>();
  if (value == "one_engine") {
    return PipelineMode::kOneEngine;
  }
  if (value == "two_stage") {
    return PipelineMode::kTwoStage;
  }
  throw std::runtime_error("Unsupported pipeline_mode: " + value);
}

}  // namespace

AppConfig LoadConfig(const std::string& path) {
  const YAML::Node root = YAML::LoadFile(path);
  AppConfig cfg;
  cfg.pipeline_mode = ParsePipelineMode(root);

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
    cfg.runtime.print_stage_timing =
        runtime["print_stage_timing"] ? runtime["print_stage_timing"].as<bool>()
                                      : cfg.runtime.print_stage_timing;
  }

  if (root["two_stage"]) {
    const auto two_stage = root["two_stage"];
    cfg.two_stage.img_engine_path =
        two_stage["img_engine_path"] ? two_stage["img_engine_path"].as<std::string>() : "";
    cfg.two_stage.bev_engine_path =
        two_stage["bev_engine_path"] ? two_stage["bev_engine_path"].as<std::string>() : "";
    cfg.two_stage.img_feature_tensor = two_stage["img_feature_tensor"]
                                           ? two_stage["img_feature_tensor"].as<std::string>()
                                           : cfg.two_stage.img_feature_tensor;
    cfg.two_stage.img_depth_tensor = two_stage["img_depth_tensor"]
                                         ? two_stage["img_depth_tensor"].as<std::string>()
                                         : cfg.two_stage.img_depth_tensor;
    cfg.two_stage.bev_input_tensor =
        two_stage["bev_input_tensor"] ? two_stage["bev_input_tensor"].as<std::string>()
                                      : cfg.two_stage.bev_input_tensor;
    cfg.two_stage.enable_bevpool_bridge =
        two_stage["enable_bevpool_bridge"] ? two_stage["enable_bevpool_bridge"].as<bool>()
                                           : cfg.two_stage.enable_bevpool_bridge;
    cfg.two_stage.use_real_bevpool =
        two_stage["use_real_bevpool"] ? two_stage["use_real_bevpool"].as<bool>()
                                      : cfg.two_stage.use_real_bevpool;
    cfg.two_stage.enable_temporal_concat =
        two_stage["enable_temporal_concat"] ? two_stage["enable_temporal_concat"].as<bool>()
                                            : cfg.two_stage.enable_temporal_concat;
    cfg.two_stage.enable_geometric_align =
        two_stage["enable_geometric_align"] ? two_stage["enable_geometric_align"].as<bool>()
                                            : cfg.two_stage.enable_geometric_align;
    cfg.two_stage.adj_num = two_stage["adj_num"] ? two_stage["adj_num"].as<int>() : cfg.two_stage.adj_num;
    cfg.two_stage.transform_matrices_path =
        two_stage["transform_matrices_path"] ? two_stage["transform_matrices_path"].as<std::string>() : "";
    cfg.two_stage.transform_sequence_dir =
        two_stage["transform_sequence_dir"] ? two_stage["transform_sequence_dir"].as<std::string>() : "";
    cfg.two_stage.ranks_depth_path =
        two_stage["ranks_depth_path"] ? two_stage["ranks_depth_path"].as<std::string>() : "";
    cfg.two_stage.ranks_feat_path =
        two_stage["ranks_feat_path"] ? two_stage["ranks_feat_path"].as<std::string>() : "";
    cfg.two_stage.ranks_bev_path =
        two_stage["ranks_bev_path"] ? two_stage["ranks_bev_path"].as<std::string>() : "";
    cfg.two_stage.interval_starts_path =
        two_stage["interval_starts_path"] ? two_stage["interval_starts_path"].as<std::string>() : "";
    cfg.two_stage.interval_lengths_path = two_stage["interval_lengths_path"]
                                              ? two_stage["interval_lengths_path"].as<std::string>()
                                              : "";
  }

  if (cfg.pipeline_mode == PipelineMode::kOneEngine && cfg.build.engine_path.empty()) {
    throw std::runtime_error("Config missing required key: build.engine_path");
  }
  if (cfg.pipeline_mode == PipelineMode::kTwoStage &&
      (cfg.two_stage.img_engine_path.empty() || cfg.two_stage.bev_engine_path.empty())) {
    throw std::runtime_error(
        "two_stage mode requires both two_stage.img_engine_path and two_stage.bev_engine_path");
  }
  if (cfg.pipeline_mode == PipelineMode::kTwoStage && cfg.two_stage.use_real_bevpool) {
    if (cfg.two_stage.ranks_depth_path.empty() || cfg.two_stage.ranks_feat_path.empty() ||
        cfg.two_stage.ranks_bev_path.empty() || cfg.two_stage.interval_starts_path.empty() ||
        cfg.two_stage.interval_lengths_path.empty()) {
      throw std::runtime_error(
          "two_stage.use_real_bevpool=true requires ranks_* and interval_* file paths");
    }
  }
  if (cfg.pipeline_mode == PipelineMode::kTwoStage && cfg.two_stage.enable_geometric_align &&
      !cfg.two_stage.enable_temporal_concat) {
    throw std::runtime_error(
        "two_stage.enable_geometric_align=true requires two_stage.enable_temporal_concat=true");
  }
  return cfg;
}

}  // namespace tio
