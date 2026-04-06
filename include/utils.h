#pragma once

#include <NvInferRuntime.h>

#include <stdexcept>
#include <string>

namespace tio {

inline int64_t Volume(const nvinfer1::Dims& dims) {
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    v *= dims.d[i];
  }
  return v;
}

inline std::size_t ElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT8:
      return 1;
    case nvinfer1::DataType::kINT32:
      return 4;
#if NV_TENSORRT_MAJOR >= 8
    case nvinfer1::DataType::kBOOL:
      return 1;
#endif
    default:
      throw std::runtime_error("Unsupported TRT data type.");
  }
}

}  // namespace tio
