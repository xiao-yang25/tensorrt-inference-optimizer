# TensorRT Inference Optimizer

`tensorrt-inference-optimizer` 是一个面向 BEVDet 推理部署的 C++/CUDA/TensorRT 工程骨架，覆盖了：

- ONNX -> TensorRT 引擎构建（FP16/INT8）
- C++ 运行时加载与推理
- 基准测试（mean/p50/p90/p99/throughput）
- INT8 校准数据与缓存流程
- plugin 接口与数值一致性校验脚本

## 主要代码结构

- `engine/`: 引擎构建与 INT8 calibrator
- `runtime/`: 绑定、buffer 管理、推理执行
- `bench/`: benchmark 可执行程序
- `plugin/`: BEVPool / Align / Gather 插件与注册入口（含 kernel/unit/integration 测试）
- `tools/`: 一键构建/运行/导出/校验脚本
- `docs/`: 环境、流程、INT8 协议、性能模板

## 脚本入口（建议）

- `tools/build.sh`: CMake 构建入口（支持 plugin / benchmark 开关）
- `tools/run.sh`: demo/benchmark 统一运行入口
- `tools/export_engine.py`: ONNX -> TensorRT 引擎导出
- `tools/int8_fp16_report.py`: INT8 vs FP16 性能与误差报告
- `tools/run_ptq_baseline.sh`: PTQ 基线一键脚本（构建 FP16/INT8 + 报告）
- `tools/run_checkpoint_trt.sh`: `checkpoint/` 两阶段 ONNX 快速 benchmark
- `tools/oneclick_ci.sh`: 一键编译+测试并产出验证报告

## 工业化工作流

一键编译与测试（含插件测试链）：

```bash
tools/oneclick_ci.sh
```

执行后会生成：

- `docs/verification_report.md`
- `reports/ci/oneclick_ci.log`

## 已验证环境（实机）

- GPU: NVIDIA GeForce RTX 3090
- Driver: 550.107.02
- CUDA Toolkit: 12.4
- TensorRT: 10.1.0 (`+cuda12.4`)
- CMake: 3.22.1
- g++: 11.4.0

## 快速开始

### A) 真实两阶段 ONNX（推荐，与本仓库 `checkpoint/` 对齐）

本仓库在 `checkpoint/` 下提供了与 [`LCH1238/BEVDet` export 分支说明](https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md)一致的产物形态：

- `checkpoint/img_stage_lt_d.onnx` / `checkpoint/bev_stage_lt_d.onnx`
- `checkpoint/img_stage_ft.onnx` / `checkpoint/bev_stage_ft.onnx`
- `checkpoint/bevdet-lt-d-ft-nearest.pth`（训练权重；两阶段 ONNX 已包含导出推理所需权重）

一键构建 FP16 engine 并跑 `trtexec` 自带 benchmark：

```bash
tools/run_checkpoint_trt.sh
```

或手动（等价）：

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

说明：

- `img_stage` 与 `bev_stage` 之间通常还存在 **视图变换 + BEV 特征构造** 等预处理/中间算子（不在 `bev_stage*.onnx` 的输入里）。因此这里的“跑通”指 **两张真实 ONNX 各自完成 TensorRT 构建与 GPU 推理计时**；端到端链路需要按 `bevdet-tensorrt-cpp` 的 C++ 工程把中间模块接起来。
- `trtexec` 默认输入为随机数据，但 **算子与权重来自真实 ONNX**；若你要用 nuScenes 等真实传感器张量，请使用 `trtexec --loadInputs=...`（需自行准备 `.pb` 输入包）。

### B) 单输入 `model/bevdet.onnx`（工程骨架 demo / benchmark）

1) （可选）生成 INT8 校准 batch（仅用于骨架 INT8 流程验证）：

```bash
python tools/generate_dummy_calib.py --out-dir sample0 --count 8 --shape 1,6,3,256,704
```

2) 编译：

```bash
tools/build.sh -p local -B -c
```

3) 导出引擎（脚本自动兼容 TRT8/TRT10 workspace 参数）：

```bash
python tools/export_engine.py \
  --onnx model/bevdet.onnx \
  --engine engine/bevdet_fp16.engine \
  --fp16 \
  --min-shapes input:1x6x3x256x704 \
  --opt-shapes input:1x6x3x256x704 \
  --max-shapes input:1x6x3x256x704
```

4) 运行 benchmark：

```bash
tools/run.sh -m benchmark -c cfgs/bench_fp16.yaml
```

5) 生成 INT8 vs FP16 报告（时延/吞吐 + 输出误差）：

```bash
cmake -S . -B build -DTIO_ENABLE_BENCHMARK=ON
cmake --build build -j --target tio_compare_engines

python tools/int8_fp16_report.py \
  --fp16-engine engine/bevdet_fp16.engine \
  --int8-engine engine/bevdet_int8.engine \
  --compare-bin build/tio_compare_engines \
  --json-out reports/int8_fp16_report.json
```

## 实测结果（当前仓库）

以下数据来自 **RTX 3090 + TensorRT 10.1.0（CUDA 12.4）**，对 `checkpoint/*_lt_d.onnx` 运行 `trtexec --fp16 --memPoolSize=workspace:4096` 的默认 benchmark（含 H2D/D2H 的 **Latency** 与不含传输的 **GPU Compute Time**）。

| Stage | GPU Compute mean(ms) | Latency mean(ms) | Throughput(qps) | Engine |
|---|---:|---:|---:|---|
| img_stage_lt_d | 4.643 | 6.056 | 214.885 | `engine/img_stage_lt_d_fp16.engine` (96M) |
| bev_stage_lt_d | 1.712 | 5.988 | 246.329 | `engine/bev_stage_lt_d_fp16.engine` (57M) |

把两阶段 **GPU 纯算**按均值相加（粗粒度上界/估算）：约 **6.355 ms / frame**（未计入中间 BEV 特征构造与其他 CPU 工作）。

## 常见问题

- CMake 报 `TensorRT not found`：检查 `NvInfer.h` / `libnvinfer.so` 是否存在，并确认 `TensorRT_ROOT`
- `trtexec: command not found`：一般在 `/usr/src/tensorrt/bin/trtexec`
- `CUDA driver version is insufficient...`：TensorRT 包版本与 CUDA/驱动不匹配（例如误装 `+cuda13.x`）
- TRT10 报 `Unknown option: --workspace`：应使用 `--memPoolSize=workspace:<MiB>`（已在脚本中兼容）

更多细节见：

- `docs/environment.md`
- `docs/runbook.md`
- `docs/int8_evaluation.md`
- `docs/architecture.md`
- `docs/resume_project_brief.md`
