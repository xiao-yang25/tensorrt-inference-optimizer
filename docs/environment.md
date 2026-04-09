# Environment Matrix

## Desktop / Server

- CUDA: 12.2+ (recommended for Ada/RTX4090: 12.4+)
- cuDNN: 8.9+ (or toolkit bundled variant)
- TensorRT: 8.6+ or 10.x (code path supports both destroy/delete API differences)
- Compiler: gcc/g++ 11+ (Ubuntu 22.04 recommended)
- CMake: 3.23+ (3.18 minimum for basic configure)
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
5. `nvcc --version` matches expected CUDA toolkit.
