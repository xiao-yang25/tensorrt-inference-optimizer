#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <vector>

#include "builder.h"
#include "config.h"
#include "infer.h"
#include "logger.h"
#include "two_stage_pipeline.h"

namespace fs = std::filesystem;

namespace {

template <typename RunFn>
bool RunLatencyLoop(const tio::RuntimeConfig& cfg, RunFn run_fn, double& mean_ms) {
  for (int i = 0; i < cfg.warmup_runs; ++i) {
    if (!run_fn()) {
      return false;
    }
  }
  std::vector<double> lat_ms;
  lat_ms.reserve(cfg.benchmark_runs);
  for (int i = 0; i < cfg.benchmark_runs; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    if (!run_fn()) {
      return false;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    lat_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  mean_ms = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / lat_ms.size();
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: tio_demo <config.yaml>" << std::endl;
    return 1;
  }

  try {
    const auto cfg = tio::LoadConfig(argv[1]);
    tio::TrtLogger logger(nvinfer1::ILogger::Severity::kINFO);

    if (cfg.pipeline_mode == tio::PipelineMode::kOneEngine && !cfg.build.onnx_path.empty() &&
        (!fs::exists(cfg.build.engine_path) || cfg.build.engine_path.empty())) {
      tio::EngineBuilder builder(&logger);
      if (!builder.BuildAndSerialize(cfg.build)) {
        std::cerr << "Failed to build engine." << std::endl;
        return 2;
      }
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      std::cerr << "Failed to create cuda stream." << std::endl;
      return 5;
    }

    double mean = 0.0;
    bool ok = false;
    if (cfg.pipeline_mode == tio::PipelineMode::kTwoStage) {
      tio::TwoStagePipeline pipeline(&logger);
      if (!pipeline.Initialize(cfg)) {
        std::cerr << "Failed to initialize two-stage pipeline." << std::endl;
        cudaStreamDestroy(stream);
        return 6;
      }
      ok = RunLatencyLoop(cfg.runtime, [&]() {
        if (!pipeline.RunOnce(stream)) {
          return false;
        }
        cudaStreamSynchronize(stream);
        return true;
      }, mean);
    } else {
      tio::InferRunner runner(&logger);
      if (!runner.LoadEngineFromFile(cfg.build.engine_path) || !runner.PrepareBindings(cfg.runtime.batch)) {
        std::cerr << "Failed to initialize one-engine runner." << std::endl;
        cudaStreamDestroy(stream);
        return 7;
      }
      ok = RunLatencyLoop(cfg.runtime, [&]() {
        if (!runner.RunOnce(stream)) {
          return false;
        }
        cudaStreamSynchronize(stream);
        return true;
      }, mean);
    }
    cudaStreamDestroy(stream);
    if (!ok) {
      std::cerr << "Inference loop failed." << std::endl;
      return 8;
    }
    std::cout << "Pipeline mode: "
              << (cfg.pipeline_mode == tio::PipelineMode::kTwoStage ? "two_stage" : "one_engine")
              << ", mean latency(ms): " << mean << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return 10;
  }
}
