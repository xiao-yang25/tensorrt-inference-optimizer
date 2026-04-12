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
  bool print_stage_timing{false};
};

enum class PipelineMode {
  kOneEngine,
  kTwoStage,
};

struct TwoStageConfig {
  std::string img_engine_path;
  std::string bev_engine_path;
  std::string img_feature_tensor{"images_feat"};
  std::string img_depth_tensor{"depth"};
  std::string bev_input_tensor{"BEV_feat"};
  bool enable_bevpool_bridge{false};
  bool use_real_bevpool{false};
  bool enable_temporal_concat{false};
  bool enable_geometric_align{false};
  int adj_num{0};
  std::string transform_matrices_path;
  std::string transform_sequence_dir;
  std::string ranks_depth_path;
  std::string ranks_feat_path;
  std::string ranks_bev_path;
  std::string interval_starts_path;
  std::string interval_lengths_path;
};

struct AppConfig {
  BuildConfig build;
  RuntimeConfig runtime;
  PipelineMode pipeline_mode{PipelineMode::kOneEngine};
  TwoStageConfig two_stage;
};

AppConfig LoadConfig(const std::string& path);

}  // namespace tio
