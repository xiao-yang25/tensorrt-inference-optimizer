# Architecture (Industrialized)

## 1. Layered Design

The project follows a clear layering model to reduce coupling and avoid duplicated logic:

- **L0 Infrastructure**
  - Build system, toolchain, third-party runtime integration
  - Main files: `CMakeLists.txt`, `scripts/build.sh`
- **L1 Engine Build**
  - ONNX parsing, precision flags, calibration wiring, profile setup
  - Main files: `engine/builder.*`, `engine/calibrator.*`
- **L2 Runtime Execution**
  - Engine loading, buffer allocation, tensor binding, stream execution
  - Main files: `runtime/infer.*`, `runtime/buffer_manager.*`
- **L3 Pipeline Logic**
  - One-engine / two-stage orchestration, temporal queue, geometric alignment
  - Main files: `runtime/two_stage_pipeline.*`
- **L4 Plugin Stack**
  - BEVPool / Align / Gather plugin kernels + TRT wrappers + registry
  - Main files: `plugin/include/*`, `plugin/src/*`, `plugin/cuda/*`, `plugin/tests/*`
- **L5 Verification & Reports**
  - Benchmarking, plugin tests, INT8-vs-FP16 comparison, CI report generation
  - Main files: `bench/*`, `scripts/int8_fp16_report.py`, `scripts/oneclick_ci.sh`

## 2. Anti-duplication Rules

- Common target include logic is centralized in CMake helper:
  - `tio_apply_target_includes(...)`
- Test registration is centralized under one CTest gate:
  - `TIO_ENABLE_TESTS=ON` + `add_test(...)`
- Script entry points are unified by responsibility:
  - Build: `scripts/build.sh`
  - Run: `scripts/run.sh`
  - Full verify: `scripts/oneclick_ci.sh`

## 3. Verification Hierarchy

The test chain is intentionally layered:

1. **Kernel Unit Tests**
   - `tio_bevpool_kernel_test`
   - `tio_alignbev_kernel_test`
   - `tio_gatherbev_kernel_test`
2. **Plugin API Smoke Test**
   - `tio_plugin_enqueue_smoke_test`
3. **TRT Graph Integration Tests**
   - `tio_plugin_minimal_trt_test`
   - `tio_plugin_chain_trt_test`
4. **Quantization Evaluation**
   - `tio_compare_engines` + `scripts/int8_fp16_report.py`

## 4. One-click Engineering Entry

Use one command for compile+test and report:

```bash
scripts/oneclick_ci.sh
```

Output artifacts:

- `output/reports/verification_report.md`
- `output/reports/ci/oneclick_ci.log`
