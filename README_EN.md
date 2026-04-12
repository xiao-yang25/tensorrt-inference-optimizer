# TensorRT Inference Optimizer

`tensorrt-inference-optimizer` is a C++/CUDA/TensorRT deployment scaffold for BEVDet-style inference, including:

- ONNX -> TensorRT engine build flow (FP16/INT8)
- C++ runtime loading and inference execution
- Benchmarking (mean/p50/p90/p99/throughput)
- INT8 calibration batch + cache workflow
- Plugin interfaces and numerical sanity-check scripts

## Project Layout

- `engine/`: engine builder and INT8 calibrator
- `runtime/`: bindings, buffers, and inference runner
- `bench/`: benchmark executable
- `plugin/`: BEVPool / Align / Gather plugins and registry (with kernel/unit/integration tests)
- `tools/`: build/run/export/validation scripts
- `docs/`: environment, pipeline, INT8 protocol, perf template

## Recommended Script Entry Points

- `tools/build.sh`: unified CMake build entry (plugin/benchmark toggles)
- `tools/run.sh`: unified runtime entry for demo/benchmark
- `tools/export_engine.py`: ONNX -> TensorRT engine export
- `tools/int8_fp16_report.py`: INT8 vs FP16 perf + output-diff report
- `tools/run_ptq_baseline.sh`: one-shot PTQ baseline (build FP16/INT8 + report)
- `tools/run_checkpoint_trt.sh`: quick benchmark for two-stage ONNX in `checkpoint/`
- `tools/oneclick_ci.sh`: one-click compile+test with verification report output

## Industrial Workflow

One-click compile and test (including plugin test chain):

```bash
tools/oneclick_ci.sh
```

Generated artifacts:

- `reports/verification_report.md`
- `reports/ci/oneclick_ci.log`

## Verified Environment (real machine)

- GPU: NVIDIA GeForce RTX 3090
- Driver: 550.107.02
- CUDA Toolkit: 12.4
- TensorRT: 10.1.0 (`+cuda12.4`)
- CMake: 3.22.1
- g++: 11.4.0

## Local artifacts (not in Git)

`checkpoint/`, `model/`, `sample0/`, and `engine/*.engine` are **not committed** to the remote. After clone, prepare them locally:

- See **`docs/local_artifacts.md`**

## Quick Start

### A) Real two-stage ONNX (recommended)

Export ONNX and weights following [`LCH1238/BEVDet` export branch README (zh-CN)](https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md), then place them under a local **`checkpoint/`** directory at the repo root, for example:

- `checkpoint/img_stage_lt_d.onnx` / `checkpoint/bev_stage_lt_d.onnx`
- `checkpoint/img_stage_ft.onnx` / `checkpoint/bev_stage_ft.onnx` (if using that variant)
- `checkpoint/bevdet-lt-d-ft-nearest.pth` (optional)

One command to build FP16 engines and run `trtexec`'s built-in benchmark:

```bash
tools/run_checkpoint_trt.sh
```

Or manually (equivalent):

```bash
mkdir -p engine

trtexec \
  --onnx=checkpoint/img_stage_lt_d.onnx \
  --saveEngine=engine/img_stage_lt_d_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096

trtexec \
  --onnx=checkpoint/bev_stage_lt_d.onnx \
  --saveEngine=engine/bev_stage_lt_d_fp16.engine \
  --fp16 \
  --memPoolSize=workspace:4096
```

Notes:

- Between `img_stage` and `bev_stage`, a real deployment still needs **view transform + BEV feature construction** (not present as inputs inside `bev_stage*.onnx`). So “end-to-end” here means **each ONNX builds and benchmarks independently** on TensorRT; wiring the middle modules is done in `bevdet-tensorrt-cpp`-style C++ pipelines.
- `trtexec` defaults to random inputs, but **ops/weights are from the real ONNX**. For dataset tensors (e.g. nuScenes), use `trtexec --loadInputs=...` with a prepared `.pb` input bundle.

### B) Single-input ONNX (scaffold demo / benchmark)

Put the ONNX (e.g. `bevdet.onnx`) under local **`model/`** and set `build.onnx_path` in your YAML (default example: `model/bevdet.onnx`).

1) (Optional) Generate INT8 calibration batches (scaffold INT8 flow only):

```bash
python tools/generate_dummy_calib.py --out-dir sample0 --count 8 --shape 1,6,3,256,704
```

2) Build:

```bash
tools/build.sh -p local -B -c
```

3) Export engine (script auto-handles TRT8/TRT10 workspace flags):

```bash
python tools/export_engine.py \
  --onnx model/bevdet.onnx \
  --engine engine/bevdet_fp16.engine \
  --fp16 \
  --min-shapes input:1x6x3x256x704 \
  --opt-shapes input:1x6x3x256x704 \
  --max-shapes input:1x6x3x256x704
```

4) Run benchmark:

```bash
tools/run.sh -m benchmark -c cfgs/default.yaml
```

## Reference benchmark (requires local `checkpoint/`)

Numbers below are from **RTX 3090 + TensorRT 10.1.0 (CUDA 12.4)**, after placing `checkpoint/img_stage_lt_d.onnx` and `checkpoint/bev_stage_lt_d.onnx` locally, running `trtexec --fp16 --memPoolSize=workspace:4096` with the default benchmark. **Latency** includes H2D + GPU + D2H; **GPU Compute Time** excludes transfers. **Not shipped with the repository**; for reproduction only.

| Stage | GPU Compute mean(ms) | Latency mean(ms) | Throughput(qps) | Engine |
|---|---:|---:|---:|---|
| img_stage_lt_d | 4.643 | 6.056 | 214.885 | `engine/img_stage_lt_d_fp16.engine` (96M) |
| bev_stage_lt_d | 1.712 | 5.988 | 246.329 | `engine/bev_stage_lt_d_fp16.engine` (57M) |

Summing GPU-compute means across stages (rough estimate): **~6.355 ms / frame**, excluding intermediate BEV construction and other CPU work.

## Common Pitfalls

- `TensorRT not found` in CMake: verify `NvInfer.h` / `libnvinfer.so` and `TensorRT_ROOT`
- `trtexec: command not found`: often installed at `/usr/src/tensorrt/bin/trtexec`
- `CUDA driver version is insufficient...`: TensorRT package variant does not match host CUDA/driver (for example `+cuda13.x` on CUDA 12.4)
- TRT10 `Unknown option: --workspace`: use `--memPoolSize=workspace:<MiB>` (handled by updated script)

See also:

- `docs/environment.md`
- `docs/runbook.md`
- `docs/int8_evaluation.md`
- `docs/architecture.md`
- `docs/directory_layout.md`
- `docs/local_artifacts.md`
- `docs/resume_project_brief.md`
