#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "config.h"
#include "infer.h"
#include "logger.h"

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

  const auto cfg = tio::LoadConfig(argv[1]);
  tio::TrtLogger logger(nvinfer1::ILogger::Severity::kWARNING);
  tio::InferRunner runner(&logger);
  if (!runner.LoadEngineFromFile(cfg.build.engine_path) || !runner.PrepareBindings(cfg.runtime.batch)) {
    std::cerr << "Failed to initialize runner." << std::endl;
    return 2;
  }

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  for (int i = 0; i < cfg.runtime.warmup_runs; ++i) {
    runner.RunOnce(stream);
  }
  cudaStreamSynchronize(stream);

  std::vector<double> lat;
  lat.reserve(cfg.runtime.benchmark_runs);
  for (int i = 0; i < cfg.runtime.benchmark_runs; ++i) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    runner.RunOnce(stream);
    cudaStreamSynchronize(stream);
    const auto t1 = std::chrono::high_resolution_clock::now();
    lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
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
  return 0;
}
