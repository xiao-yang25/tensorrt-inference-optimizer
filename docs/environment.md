# Environment Matrix

## Desktop / Server

- CUDA: 12.2+ (recommended for Ada/RTX4090: 12.4+)
- cuDNN: 8.9+ (or toolkit bundled variant)
- TensorRT: 8.6+ or 10.x (code path supports both destroy/delete API differences)
- Compiler: gcc/g++ 11+ (Ubuntu 22.04 recommended)
- CMake: 3.23+ (3.18 minimum for basic configure)
- yaml-cpp: 0.7+

## Version Matching Notes (important)

- TensorRT package variant must match your CUDA/driver capability.
- If `trtexec` prints `CUDA driver version is insufficient for CUDA runtime version`, you likely installed a too-new TensorRT build (for example `+cuda13.x`) on a CUDA 12.4 host.
- On Ubuntu 22.04 + CUDA 12.4, use TensorRT packages with suffix `+cuda12.4` (for example `10.1.0.27-1+cuda12.4`), then re-run build.
- `trtexec` may be installed at `/usr/src/tensorrt/bin/trtexec` by apt packages. Add it to PATH (for example symlink to `/usr/local/bin/trtexec`) if command is not found.

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

## Proven Setup on This Repo (Apr 2026)

- GPU: NVIDIA GeForce RTX 3090
- Driver: 550.107.02
- CUDA Toolkit: 12.4
- TensorRT: 10.1.0 (`+cuda12.4`)
- CMake: 3.22.1
- g++: 11.4.0
- yaml-cpp: 0.7.x

This setup successfully built:

- `build/tio_demo`
- `build/tio_benchmark`
