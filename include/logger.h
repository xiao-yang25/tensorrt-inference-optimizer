#pragma once

#include <NvInferRuntime.h>

#include <iostream>
#include <mutex>
#include <string>

namespace tio {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  explicit TrtLogger(Severity min_severity = Severity::kWARNING)
      : min_severity_(min_severity) {}

  void log(Severity severity, const char* msg) noexcept override {
    if (severity > min_severity_) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    std::cerr << "[TensorRT][" << SeverityToString(severity) << "] " << msg << std::endl;
  }

 private:
  static std::string SeverityToString(Severity severity) {
    switch (severity) {
      case Severity::kINTERNAL_ERROR:
        return "INTERNAL_ERROR";
      case Severity::kERROR:
        return "ERROR";
      case Severity::kWARNING:
        return "WARNING";
      case Severity::kINFO:
        return "INFO";
      case Severity::kVERBOSE:
        return "VERBOSE";
      default:
        return "UNKNOWN";
    }
  }

  Severity min_severity_;
  std::mutex mu_;
};

}  // namespace tio
