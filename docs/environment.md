# Environment Matrix

## Desktop / Server

- CUDA: 11.8+
- cuDNN: 8.6+
- TensorRT: 8.5.2.2+ (tested API path)
- Compiler: gcc/g++ 9+ or MSVC with CUDA support
- CMake: 3.18+
- yaml-cpp: 0.7+

## Jetson AGX Orin

- Jetpack: 5.1.1
- CUDA: 11.4+
- cuDNN: 8.6+
- TensorRT: 8.5.2.2+

## Validation Checklist

1. `nvidia-smi` or `tegrastats` available.
2. `trtexec --version` available.
3. TensorRT headers include `NvInfer.h` and `NvOnnxParser.h`.
4. `cmake --version` >= 3.18.
