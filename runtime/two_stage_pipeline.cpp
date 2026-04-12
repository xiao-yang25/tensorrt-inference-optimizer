#include "two_stage_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "bevpool_cuda.h"
#include "bev_align_cuda.h"

namespace {

std::vector<int32_t> LoadInt32BinaryFile(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    return {};
  }
  ifs.seekg(0, std::ios::end);
  const std::streamoff size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  if (size <= 0 || size % static_cast<std::streamoff>(sizeof(int32_t)) != 0) {
    return {};
  }
  std::vector<int32_t> out(static_cast<std::size_t>(size / sizeof(int32_t)));
  ifs.read(reinterpret_cast<char*>(out.data()), size);
  if (!ifs.good() && !ifs.eof()) {
    return {};
  }
  return out;
}

}  // namespace

namespace tio {

TwoStagePipeline::TwoStagePipeline(TrtLogger* logger)
    : logger_(logger), img_runner_(logger), bev_runner_(logger) {}

TwoStagePipeline::~TwoStagePipeline() {
  ReleaseBevPoolIndices();
  ReleaseTemporalBuffers();
}

bool TwoStagePipeline::Initialize(const AppConfig& cfg) {
  runtime_cfg_ = cfg.runtime;
  cfg_ = cfg.two_stage;
  if (!img_runner_.LoadEngineFromFile(cfg_.img_engine_path)) {
    std::cerr << "Failed to load image-stage engine: " << cfg_.img_engine_path << "\n";
    return false;
  }
  if (!bev_runner_.LoadEngineFromFile(cfg_.bev_engine_path)) {
    std::cerr << "Failed to load bev-stage engine: " << cfg_.bev_engine_path << "\n";
    return false;
  }
  if (!img_runner_.PrepareBindings(cfg.runtime.batch)) {
    std::cerr << "Failed to allocate image-stage bindings.\n";
    return false;
  }
  if (!bev_runner_.PrepareBindings(cfg.runtime.batch)) {
    std::cerr << "Failed to allocate bev-stage bindings.\n";
    return false;
  }
  const auto* img_feat = img_runner_.Buffers().GetBinding(cfg_.img_feature_tensor);
  const auto* bev_in = bev_runner_.Buffers().GetBinding(cfg_.bev_input_tensor);
  if (!img_feat || !bev_in || img_feat->dims.nbDims < 1 || bev_in->dims.nbDims < 4) {
    std::cerr << "Unable to infer BEV tensor shape from bindings.\n";
    return false;
  }
  bevpool_channels_ = img_feat->dims.d[img_feat->dims.nbDims - 1];
  bev_input_channels_ = bev_in->dims.d[1];
  bev_map_size_ = bev_in->dims.d[bev_in->dims.nbDims - 1] * bev_in->dims.d[bev_in->dims.nbDims - 2];
  if (cfg_.enable_temporal_concat && !InitializeTemporalBuffers()) {
    std::cerr << "Failed to initialize temporal BEV buffers.\n";
    return false;
  }
  if (cfg_.use_real_bevpool && !InitializeBevPoolIndices()) {
    std::cerr << "Failed to initialize real BEVPool index tensors.\n";
    return false;
  }
  initialized_ = true;
  return true;
}

bool TwoStagePipeline::RunOnce(cudaStream_t stream) {
  if (!initialized_) {
    return false;
  }
  const auto total_start = std::chrono::high_resolution_clock::now();
  const auto img_start = total_start;
  if (!img_runner_.RunOnce(stream)) {
    return false;
  }
  cudaStreamSynchronize(stream);
  const auto img_end = std::chrono::high_resolution_clock::now();
  if (!BridgeToBev(stream)) {
    return false;
  }
  cudaStreamSynchronize(stream);
  const auto bevpool_end = std::chrono::high_resolution_clock::now();

  if (!RunTemporalConcat(stream)) {
    return false;
  }
  cudaStreamSynchronize(stream);
  const auto align_end = std::chrono::high_resolution_clock::now();

  if (!bev_runner_.RunOnce(stream)) {
    return false;
  }
  cudaStreamSynchronize(stream);
  const auto bev_end = std::chrono::high_resolution_clock::now();

  last_timing_.img_ms = std::chrono::duration<double, std::milli>(img_end - img_start).count();
  last_timing_.bevpool_ms =
      std::chrono::duration<double, std::milli>(bevpool_end - img_end).count();
  last_timing_.align_ms = std::chrono::duration<double, std::milli>(align_end - bevpool_end).count();
  last_timing_.bev_ms = std::chrono::duration<double, std::milli>(bev_end - align_end).count();
  last_timing_.total_ms = std::chrono::duration<double, std::milli>(bev_end - total_start).count();
  return true;
}

bool TwoStagePipeline::BridgeToBev(cudaStream_t stream) {
  const auto* img_feat = img_runner_.Buffers().GetBinding(cfg_.img_feature_tensor);
  const auto* bev_in = bev_runner_.Buffers().GetBinding(cfg_.bev_input_tensor);
  if (!img_feat || !bev_in) {
    std::cerr << "Bridge tensor missing. img_feature=" << cfg_.img_feature_tensor
              << ", bev_input=" << cfg_.bev_input_tensor << "\n";
    return false;
  }

  if (bevpool_ready_) {
    const auto* img_depth = img_runner_.Buffers().GetBinding(cfg_.img_depth_tensor);
    if (!img_depth) {
      std::cerr << "Real BEVPool enabled but depth tensor not found: " << cfg_.img_depth_tensor << "\n";
      return false;
    }
    const int channels = bevpool_channels_;
    const int map_size = bev_map_size_;
    const int bev_channels = bev_input_channels_;
    if (channels <= 0 || map_size <= 0 || bev_channels < channels) {
      std::cerr << "Invalid BEVPool shape values: channels=" << channels
                << ", bev_channels=" << bev_channels << ", map_size=" << map_size << "\n";
      return false;
    }
    cudaMemsetAsync(bev_in->ptr, 0, bev_in->bytes, stream);
    LaunchBevPoolV2(channels, n_intervals_, map_size, static_cast<const float*>(img_depth->ptr),
                    static_cast<const float*>(img_feat->ptr), ranks_depth_dev_, ranks_feat_dev_, ranks_bev_dev_,
                    interval_starts_dev_, interval_lengths_dev_, static_cast<float*>(bev_in->ptr), stream);
    return cudaPeekAtLastError() == cudaSuccess;
  }

  // Transitional scaffold path when real BEVPool indices are unavailable.
  if (cfg_.enable_bevpool_bridge) {
    const auto bytes = std::min(img_feat->bytes, bev_in->bytes);
    cudaMemsetAsync(bev_in->ptr, 0, bev_in->bytes, stream);
    cudaMemcpyAsync(bev_in->ptr, img_feat->ptr, bytes, cudaMemcpyDeviceToDevice, stream);
    return cudaPeekAtLastError() == cudaSuccess;
  }

  if (img_feat->bytes != bev_in->bytes) {
    std::cerr << "Direct bridge disabled and tensor bytes mismatch: img_feature=" << img_feat->bytes
              << ", bev_input=" << bev_in->bytes
              << ". Enable two_stage.enable_bevpool_bridge for scaffold copy mode.\n";
    return false;
  }
  cudaMemcpyAsync(bev_in->ptr, img_feat->ptr, bev_in->bytes, cudaMemcpyDeviceToDevice, stream);
  return cudaPeekAtLastError() == cudaSuccess;
}

bool TwoStagePipeline::InitializeBevPoolIndices() {
  const auto ranks_depth = LoadInt32BinaryFile(cfg_.ranks_depth_path);
  const auto ranks_feat = LoadInt32BinaryFile(cfg_.ranks_feat_path);
  const auto ranks_bev = LoadInt32BinaryFile(cfg_.ranks_bev_path);
  const auto interval_starts = LoadInt32BinaryFile(cfg_.interval_starts_path);
  const auto interval_lengths = LoadInt32BinaryFile(cfg_.interval_lengths_path);
  if (ranks_depth.empty() || ranks_feat.empty() || ranks_bev.empty() || interval_starts.empty() ||
      interval_lengths.empty()) {
    std::cerr << "Failed to load BEVPool index binaries.\n";
    return false;
  }
  if (ranks_depth.size() != ranks_feat.size() || ranks_depth.size() != ranks_bev.size()) {
    std::cerr << "ranks_depth/ranks_feat/ranks_bev size mismatch.\n";
    return false;
  }
  if (interval_starts.size() != interval_lengths.size()) {
    std::cerr << "interval_starts/interval_lengths size mismatch.\n";
    return false;
  }

  const auto alloc_copy = [](const std::vector<int32_t>& host, int** dev) -> bool {
    const auto bytes = host.size() * sizeof(int32_t);
    if (cudaMalloc(reinterpret_cast<void**>(dev), bytes) != cudaSuccess) {
      return false;
    }
    if (cudaMemcpy(*dev, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
    return true;
  };
  if (!alloc_copy(ranks_depth, &ranks_depth_dev_) || !alloc_copy(ranks_feat, &ranks_feat_dev_) ||
      !alloc_copy(ranks_bev, &ranks_bev_dev_) || !alloc_copy(interval_starts, &interval_starts_dev_) ||
      !alloc_copy(interval_lengths, &interval_lengths_dev_)) {
    std::cerr << "Failed to upload BEVPool index tensors to device.\n";
    ReleaseBevPoolIndices();
    return false;
  }

  n_intervals_ = static_cast<int>(interval_lengths.size());
  bevpool_ready_ = true;
  return true;
}

bool TwoStagePipeline::InitializeTemporalBuffers() {
  if (!cfg_.enable_temporal_concat || cfg_.adj_num <= 0) {
    return true;
  }
  if (bevpool_channels_ <= 0 || bev_map_size_ <= 0 || bev_input_channels_ <= 0) {
    std::cerr << "Invalid BEV tensor shape for temporal concat.\n";
    return false;
  }
  if (bev_input_channels_ < bevpool_channels_ * (cfg_.adj_num + 1)) {
    std::cerr << "BEV input channels not enough for temporal concat: need "
              << bevpool_channels_ * (cfg_.adj_num + 1) << ", got " << bev_input_channels_ << "\n";
    return false;
  }
  const std::size_t bytes = static_cast<std::size_t>(bevpool_channels_) * bev_map_size_ * sizeof(float);
  temporal_bev_buffers_.resize(cfg_.adj_num, nullptr);
  geometric_transforms_dev_.resize(cfg_.adj_num, nullptr);
  geometric_grids_dev_.resize(cfg_.adj_num, nullptr);
  for (int i = 0; i < cfg_.adj_num; ++i) {
    if (cudaMalloc(&temporal_bev_buffers_[i], bytes) != cudaSuccess) {
      std::cerr << "Failed to allocate temporal BEV buffer " << i << "\n";
      ReleaseTemporalBuffers();
      return false;
    }
    cudaMemset(temporal_bev_buffers_[i], 0, bytes);
    if (cfg_.enable_geometric_align) {
      if (cudaMalloc(&geometric_transforms_dev_[i], 9 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate geometric transform buffer " << i << "\n";
        ReleaseTemporalBuffers();
        return false;
      }
      if (cudaMalloc(&geometric_grids_dev_[i], bev_map_size_ * 2 * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate geometric grid buffer " << i << "\n";
        ReleaseTemporalBuffers();
        return false;
      }
    }
  }
  if (cfg_.enable_geometric_align && !LoadGeometricTransforms()) {
    std::cerr << "Failed to load geometric transform matrices.\n";
    ReleaseTemporalBuffers();
    return false;
  }
  if (cfg_.enable_geometric_align && !LoadTransformSequence()) {
    std::cerr << "Failed to load transform sequence.\n";
    ReleaseTemporalBuffers();
    return false;
  }
  temporal_primed_ = false;
  return true;
}

bool TwoStagePipeline::LoadGeometricTransforms() {
  geometric_transforms_host_.clear();
  geometric_transforms_host_.resize(static_cast<std::size_t>(cfg_.adj_num) * 9U, 0.0F);
  for (int i = 0; i < cfg_.adj_num; ++i) {
    geometric_transforms_host_[i * 9 + 0] = 1.0F;
    geometric_transforms_host_[i * 9 + 4] = 1.0F;
    geometric_transforms_host_[i * 9 + 8] = 1.0F;
  }
  if (!cfg_.transform_matrices_path.empty()) {
    std::ifstream ifs(cfg_.transform_matrices_path, std::ios::binary);
    if (!ifs) {
      std::cerr << "Unable to read transform_matrices_path: " << cfg_.transform_matrices_path << "\n";
      return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    const std::size_t expected = static_cast<std::size_t>(cfg_.adj_num) * 9U * sizeof(float);
    if (size != static_cast<std::streamoff>(expected)) {
      std::cerr << "transform_matrices_path size mismatch. expected bytes=" << expected
                << ", got=" << size << "\n";
      return false;
    }
    ifs.read(reinterpret_cast<char*>(geometric_transforms_host_.data()), size);
  }
  for (int i = 0; i < cfg_.adj_num; ++i) {
    if (cudaMemcpy(geometric_transforms_dev_[i], geometric_transforms_host_.data() + i * 9, 9 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
  }
  return true;
}

bool TwoStagePipeline::LoadTransformSequence() {
  geometric_transform_sequence_.clear();
  if (cfg_.transform_sequence_dir.empty()) {
    return true;
  }
  namespace fs = std::filesystem;
  std::vector<fs::path> files;
  for (const auto& entry : fs::directory_iterator(cfg_.transform_sequence_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    if (entry.path().extension() == ".bin") {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  if (files.empty()) {
    std::cerr << "No *.bin transform files found in: " << cfg_.transform_sequence_dir << "\n";
    return false;
  }
  const std::size_t expected = static_cast<std::size_t>(cfg_.adj_num) * 9U * sizeof(float);
  for (const auto& p : files) {
    std::ifstream ifs(p, std::ios::binary);
    if (!ifs) {
      std::cerr << "Unable to read transform sequence file: " << p << "\n";
      return false;
    }
    ifs.seekg(0, std::ios::end);
    const std::streamoff size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    if (size != static_cast<std::streamoff>(expected)) {
      std::cerr << "Transform sequence file size mismatch: " << p << "\n";
      return false;
    }
    std::vector<float> frame(static_cast<std::size_t>(cfg_.adj_num) * 9U);
    ifs.read(reinterpret_cast<char*>(frame.data()), size);
    geometric_transform_sequence_.push_back(std::move(frame));
  }
  transform_frame_index_ = 0;
  return true;
}

bool TwoStagePipeline::UploadTransformsForFrame(std::size_t frame_index) {
  if (geometric_transform_sequence_.empty()) {
    return true;
  }
  const auto& frame = geometric_transform_sequence_[frame_index % geometric_transform_sequence_.size()];
  for (int i = 0; i < cfg_.adj_num; ++i) {
    if (cudaMemcpy(geometric_transforms_dev_[i], frame.data() + i * 9, 9 * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }
  }
  return true;
}

bool TwoStagePipeline::RunTemporalConcat(cudaStream_t stream) {
  if (!cfg_.enable_temporal_concat || cfg_.adj_num <= 0) {
    return true;
  }
  auto* bev_in = bev_runner_.Buffers().GetBinding(cfg_.bev_input_tensor);
  if (!bev_in) {
    std::cerr << "Temporal concat requires bev input tensor: " << cfg_.bev_input_tensor << "\n";
    return false;
  }
  const std::size_t one_bev_bytes =
      static_cast<std::size_t>(bevpool_channels_) * bev_map_size_ * sizeof(float);
  auto* bev_ptr = static_cast<float*>(bev_in->ptr);
  const int bev_h = bev_in->dims.d[bev_in->dims.nbDims - 1];
  const int bev_w = bev_in->dims.d[bev_in->dims.nbDims - 2];
  if (cfg_.enable_geometric_align) {
    if (!UploadTransformsForFrame(transform_frame_index_)) {
      std::cerr << "Failed to upload transforms for frame " << transform_frame_index_ << "\n";
      return false;
    }
    transform_frame_index_++;
  }
  if (!temporal_primed_) {
    for (int i = 0; i < cfg_.adj_num; ++i) {
      cudaMemcpyAsync(bev_ptr + (i + 1) * bevpool_channels_ * bev_map_size_, bev_ptr, one_bev_bytes,
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(temporal_bev_buffers_[i], bev_ptr, one_bev_bytes, cudaMemcpyDeviceToDevice, stream);
    }
    temporal_primed_ = true;
    return cudaPeekAtLastError() == cudaSuccess;
  }
  for (int i = 0; i < cfg_.adj_num; ++i) {
    float* out_slot = bev_ptr + (i + 1) * bevpool_channels_ * bev_map_size_;
    if (cfg_.enable_geometric_align) {
      LaunchComputeSampleGrid(static_cast<float*>(geometric_grids_dev_[i]),
                              static_cast<float*>(geometric_transforms_dev_[i]), bev_w, bev_h, stream);
      LaunchGridSampleBilinear(out_slot, static_cast<const float*>(temporal_bev_buffers_[i]),
                               static_cast<const float*>(geometric_grids_dev_[i]), bevpool_channels_, bev_w,
                               bev_h, stream);
    } else {
      cudaMemcpyAsync(out_slot, temporal_bev_buffers_[i], one_bev_bytes, cudaMemcpyDeviceToDevice, stream);
    }
  }
  for (int i = cfg_.adj_num - 1; i > 0; --i) {
    cudaMemcpyAsync(temporal_bev_buffers_[i], temporal_bev_buffers_[i - 1], one_bev_bytes,
                    cudaMemcpyDeviceToDevice, stream);
  }
  cudaMemcpyAsync(temporal_bev_buffers_[0], bev_ptr, one_bev_bytes, cudaMemcpyDeviceToDevice, stream);
  return cudaPeekAtLastError() == cudaSuccess;
}

void TwoStagePipeline::ReleaseBevPoolIndices() {
  if (ranks_depth_dev_) cudaFree(ranks_depth_dev_);
  if (ranks_feat_dev_) cudaFree(ranks_feat_dev_);
  if (ranks_bev_dev_) cudaFree(ranks_bev_dev_);
  if (interval_starts_dev_) cudaFree(interval_starts_dev_);
  if (interval_lengths_dev_) cudaFree(interval_lengths_dev_);
  ranks_depth_dev_ = nullptr;
  ranks_feat_dev_ = nullptr;
  ranks_bev_dev_ = nullptr;
  interval_starts_dev_ = nullptr;
  interval_lengths_dev_ = nullptr;
  n_intervals_ = 0;
  bevpool_ready_ = false;
}

void TwoStagePipeline::ReleaseTemporalBuffers() {
  for (void*& ptr : temporal_bev_buffers_) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
  for (void*& ptr : geometric_transforms_dev_) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
  for (void*& ptr : geometric_grids_dev_) {
    if (ptr) {
      cudaFree(ptr);
      ptr = nullptr;
    }
  }
  temporal_bev_buffers_.clear();
  geometric_transforms_dev_.clear();
  geometric_grids_dev_.clear();
  geometric_transforms_host_.clear();
  geometric_transform_sequence_.clear();
  transform_frame_index_ = 0;
  temporal_primed_ = false;
}

}  // namespace tio
