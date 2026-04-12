#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"
#include "infer.h"
#include "logger.h"
#include "two_stage_pipeline.h"

namespace {

double Percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto idx = static_cast<std::size_t>(p * (values.size() - 1));
  return values[idx];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: tio_benchmark <config.yaml>" << std::endl;
    return 1;
  }

  try {
    const auto cfg = tio::LoadConfig(argv[1]);
    tio::TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    std::vector<double> lat;
    lat.reserve(cfg.runtime.benchmark_runs);
    auto run_record = [&](auto&& fn, auto&& after_benchmark_iter) -> bool {
      for (int i = 0; i < cfg.runtime.warmup_runs; ++i) {
        if (!fn()) {
          return false;
        }
      }
      cudaStreamSynchronize(stream);
      for (int i = 0; i < cfg.runtime.benchmark_runs; ++i) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        if (!fn()) {
          return false;
        }
        cudaStreamSynchronize(stream);
        const auto t1 = std::chrono::high_resolution_clock::now();
        lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        after_benchmark_iter();
      }
      return true;
    };

    if (cfg.pipeline_mode == tio::PipelineMode::kTwoStage) {
      tio::TwoStagePipeline pipeline(&logger);
      if (!pipeline.Initialize(cfg)) {
        std::cerr << "Failed to initialize two-stage pipeline." << std::endl;
        cudaStreamDestroy(stream);
        return 2;
      }
      double stage_img = 0.0;
      double stage_bevpool = 0.0;
      double stage_align = 0.0;
      double stage_bev = 0.0;
      double stage_total = 0.0;
      if (!run_record([&]() { return pipeline.RunOnce(stream); }, [&]() {
            if (!cfg.runtime.print_stage_timing) {
              return;
            }
            const auto& t = pipeline.LastTiming();
            stage_img += t.img_ms;
            stage_bevpool += t.bevpool_ms;
            stage_align += t.align_ms;
            stage_bev += t.bev_ms;
            stage_total += t.total_ms;
          })) {
        std::cerr << "Two-stage run failed." << std::endl;
        cudaStreamDestroy(stream);
        return 3;
      }
      if (cfg.runtime.print_stage_timing) {
        const double denom = static_cast<double>(cfg.runtime.benchmark_runs);
        std::cout << "stage_mean(ms): img=" << stage_img / denom
                  << ", bevpool=" << stage_bevpool / denom << ", align=" << stage_align / denom
                  << ", bev=" << stage_bev / denom << ", total=" << stage_total / denom << "\n";
      }
    } else {
      tio::InferRunner runner(&logger);
      if (!runner.LoadEngineFromFile(cfg.build.engine_path) || !runner.PrepareBindings(cfg.runtime.batch)) {
        std::cerr << "Failed to initialize one-engine runner." << std::endl;
        cudaStreamDestroy(stream);
        return 2;
      }
      if (!run_record([&]() { return runner.RunOnce(stream); }, [&]() {})) {
        std::cerr << "One-engine run failed." << std::endl;
        cudaStreamDestroy(stream);
        return 3;
      }
    }
    cudaStreamDestroy(stream);

    const auto mean = std::accumulate(lat.begin(), lat.end(), 0.0) / static_cast<double>(lat.size());
    const auto p50 = Percentile(lat, 0.50);
    const auto p90 = Percentile(lat, 0.90);
    const auto p99 = Percentile(lat, 0.99);
    const auto throughput = 1000.0 * static_cast<double>(cfg.runtime.batch) / mean;

    std::cout << "mean(ms): " << mean << "\n";
    std::cout << "p50(ms): " << p50 << ", p90(ms): " << p90 << ", p99(ms): " << p99 << "\n";
    std::cout << "throughput(samples/s): " << throughput << std::endl;
    std::cout << "pipeline_mode: "
              << (cfg.pipeline_mode == tio::PipelineMode::kTwoStage ? "two_stage" : "one_engine")
              << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return 10;
  }
}
