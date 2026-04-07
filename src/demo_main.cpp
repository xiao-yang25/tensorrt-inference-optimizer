#include <cuda_runtime_api.h>

#include <chrono>
#include <filesystem>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "builder.h"
#include "config.h"
#include "infer.h"
#include "logger.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: tio_demo <config.yaml>" << std::endl;
    return 1;
  }

  try {
    const auto cfg = tio::LoadConfig(argv[1]);
    tio::TrtLogger logger(nvinfer1::ILogger::Severity::kINFO);

    if (!cfg.build.onnx_path.empty() && (!fs::exists(cfg.build.engine_path) || cfg.build.engine_path.empty())) {
      tio::EngineBuilder builder(&logger);
      if (!builder.BuildAndSerialize(cfg.build)) {
        std::cerr << "Failed to build engine." << std::endl;
        return 2;
      }
    }

    tio::InferRunner runner(&logger);
    if (!runner.LoadEngineFromFile(cfg.build.engine_path)) {
      std::cerr << "Failed to load engine from: " << cfg.build.engine_path << std::endl;
      return 3;
    }
    if (!runner.PrepareBindings(cfg.runtime.batch)) {
      std::cerr << "Failed to prepare device buffers." << std::endl;
      return 4;
    }

    cudaStream_t h2d_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaStream_t d2h_stream = nullptr;
    if (cudaStreamCreate(&h2d_stream) != cudaSuccess || cudaStreamCreate(&compute_stream) != cudaSuccess ||
        cudaStreamCreate(&d2h_stream) != cudaSuccess) {
      std::cerr << "Failed to create cuda streams." << std::endl;
      return 5;
    }

    cudaEvent_t h2d_done = nullptr;
    cudaEvent_t compute_done = nullptr;
    cudaEventCreate(&h2d_done);
    cudaEventCreate(&compute_done);

    std::vector<void*> host_staging(runner.Buffers().DeviceBindings().size(), nullptr);
    for (std::size_t i = 0; i < runner.Buffers().DeviceBindings().size(); ++i) {
      const auto& b = runner.Buffers().DeviceBindings()[i];
      if (cudaHostAlloc(&host_staging[i], b.bytes, cudaHostAllocPortable) != cudaSuccess) {
        std::cerr << "Failed to allocate pinned host memory." << std::endl;
        return 5;
      }
      std::memset(host_staging[i], 0, b.bytes);
    }

    auto run_one = [&runner, &host_staging, h2d_stream, compute_stream, d2h_stream, h2d_done,
                    compute_done]() -> bool {
      auto& bindings = runner.Buffers().DeviceBindings();
      for (std::size_t i = 0; i < bindings.size(); ++i) {
        if (!bindings[i].is_input) {
          continue;
        }
        cudaMemcpyAsync(bindings[i].ptr, host_staging[i], bindings[i].bytes, cudaMemcpyHostToDevice, h2d_stream);
      }
      cudaEventRecord(h2d_done, h2d_stream);
      cudaStreamWaitEvent(compute_stream, h2d_done, 0);
      if (!runner.RunOnce(compute_stream)) {
        return false;
      }
      cudaEventRecord(compute_done, compute_stream);
      cudaStreamWaitEvent(d2h_stream, compute_done, 0);

      for (std::size_t i = 0; i < bindings.size(); ++i) {
        if (bindings[i].is_input) {
          continue;
        }
        cudaMemcpyAsync(host_staging[i], bindings[i].ptr, bindings[i].bytes, cudaMemcpyDeviceToHost, d2h_stream);
      }
      cudaStreamSynchronize(d2h_stream);
      return true;
    };

    for (int i = 0; i < cfg.runtime.warmup_runs; ++i) {
      if (!run_one()) {
        std::cerr << "Warmup enqueue failed." << std::endl;
        cudaStreamDestroy(h2d_stream);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(d2h_stream);
        return 6;
      }
    }

    std::vector<double> lat_ms;
    lat_ms.reserve(cfg.runtime.benchmark_runs);
    for (int i = 0; i < cfg.runtime.benchmark_runs; ++i) {
      const auto start = std::chrono::high_resolution_clock::now();
      if (!run_one()) {
        std::cerr << "Benchmark enqueue failed at iter: " << i << std::endl;
        cudaStreamDestroy(h2d_stream);
        cudaStreamDestroy(compute_stream);
        cudaStreamDestroy(d2h_stream);
        return 7;
      }
      const auto end = std::chrono::high_resolution_clock::now();
      lat_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    for (void*& ptr : host_staging) {
      if (ptr) {
        cudaFreeHost(ptr);
      }
    }
    cudaEventDestroy(h2d_done);
    cudaEventDestroy(compute_done);
    cudaStreamDestroy(h2d_stream);
    cudaStreamDestroy(compute_stream);
    cudaStreamDestroy(d2h_stream);

    const double mean = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / lat_ms.size();
    std::cout << "Benchmark runs: " << lat_ms.size() << ", mean latency(ms): " << mean << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Unhandled exception: " << e.what() << std::endl;
    return 10;
  }
}
