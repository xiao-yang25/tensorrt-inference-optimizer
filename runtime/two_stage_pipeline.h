#pragma once

#include <cuda_runtime_api.h>

#include <string>
#include <vector>

#include "config.h"
#include "infer.h"
#include "logger.h"

namespace tio {

struct TwoStageTiming {
  double img_ms{0.0};
  double bevpool_ms{0.0};
  double align_ms{0.0};
  double bev_ms{0.0};
  double total_ms{0.0};
};

class TwoStagePipeline {
 public:
  explicit TwoStagePipeline(TrtLogger* logger);
  ~TwoStagePipeline();

  bool Initialize(const AppConfig& cfg);
  bool RunOnce(cudaStream_t stream);
  const TwoStageTiming& LastTiming() const { return last_timing_; }

  InferRunner& ImgRunner() { return img_runner_; }
  InferRunner& BevRunner() { return bev_runner_; }

 private:
  bool InitializeBevPoolIndices();
  bool InitializeTemporalBuffers();
  bool RunTemporalConcat(cudaStream_t stream);
  bool LoadGeometricTransforms();
  bool LoadTransformSequence();
  bool UploadTransformsForFrame(std::size_t frame_index);
  bool BridgeToBev(cudaStream_t stream);
  void ReleaseBevPoolIndices();
  void ReleaseTemporalBuffers();

  TrtLogger* logger_{nullptr};
  RuntimeConfig runtime_cfg_{};
  TwoStageConfig cfg_{};
  InferRunner img_runner_;
  InferRunner bev_runner_;
  bool initialized_{false};
  int* ranks_depth_dev_{nullptr};
  int* ranks_feat_dev_{nullptr};
  int* ranks_bev_dev_{nullptr};
  int* interval_starts_dev_{nullptr};
  int* interval_lengths_dev_{nullptr};
  int n_intervals_{0};
  bool bevpool_ready_{false};
  int bevpool_channels_{0};
  int bev_input_channels_{0};
  int bev_map_size_{0};
  std::vector<void*> temporal_bev_buffers_;
  std::vector<float> geometric_transforms_host_;
  std::vector<std::vector<float>> geometric_transform_sequence_;
  std::vector<void*> geometric_transforms_dev_;
  std::vector<void*> geometric_grids_dev_;
  bool temporal_primed_{false};
  std::size_t transform_frame_index_{0};
  TwoStageTiming last_timing_{};
};

}  // namespace tio
