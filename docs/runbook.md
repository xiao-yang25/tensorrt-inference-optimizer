# Runbook

## 0) 真实 `checkpoint/` 两阶段 ONNX（TensorRT）

在本地准备的 `checkpoint/` 下放置 `img_stage*.onnx` / `bev_stage*.onnx`（与 [`LCH1238/BEVDet` export 分支说明](https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md)一致），用于 `bevdet-tensorrt-cpp` 风格部署时的分段推理；**该目录不入库**，见 `docs/local_artifacts.md`。

一键构建 FP16 engine 并跑 `trtexec` benchmark：

```bash
tools/run_checkpoint_trt.sh
```

说明：

- 这验证的是 **每张 ONNX 独立** 的 TensorRT 解析/构建/推理性能；`img_stage` → `bev_stage` 之间的 BEV 特征构造通常仍在 C++ 侧完成。
- 若需要真实数据集输入，请使用 `trtexec --loadInputs=...`（准备输入 `.pb`）。

仓库目录职责与 Git 边界见：`docs/directory_layout.md`。

## 0.2 One-click compile + test

```bash
tools/oneclick_ci.sh
```

输出：

- `reports/verification_report.md`
- `reports/ci/oneclick_ci.log`

## 0.1 Two-stage runtime modes in `tio_demo` / `tio_benchmark`

`configure.yaml` 支持两种 two-stage 连接方式：

- `two_stage.enable_bevpool_bridge: true`
  - 过渡模式，使用 copy/scaffold 连接，不执行真实 BEVPool
- `two_stage.use_real_bevpool: true`
  - 真实 BEVPool kernel 模式（已集成 `bev_pool_v2`）
  - 需要提供以下二进制 `int32` 索引文件路径：
    - `ranks_depth_path`
    - `ranks_feat_path`
    - `ranks_bev_path`
    - `interval_starts_path`
    - `interval_lengths_path`

未提供上述索引文件时，请保持 `use_real_bevpool: false`，否则配置加载会报错。

生成索引文件（复用 `bevdet-tensorrt-cpp` 的配置与样本）：

```bash
python tools/generate_bevpool_indices.py \
  --model-cfg /data/bevdet-tensorrt-cpp/cfgs/bevdet_lt_depth.yaml \
  --cam-yaml /data/bevdet-tensorrt-cpp/sample0/sample0000.yaml \
  --out-dir sample0/bevpool_indices
```

然后把 `two_stage.use_real_bevpool` 设为 `true`，并填入：

- `two_stage.ranks_depth_path`
- `two_stage.ranks_feat_path`
- `two_stage.ranks_bev_path`
- `two_stage.interval_starts_path`
- `two_stage.interval_lengths_path`

可选：开启简化版时序拼接（phase-1）

- `two_stage.enable_temporal_concat: true`
- `two_stage.adj_num: 8`

说明：当前实现为“历史 BEV 特征拼接缓存”，用于打通工程路径；后续可替换为几何对齐版本（grid_sample + ego pose 变换）。

可选：开启几何对齐（phase-1.5）

- `two_stage.enable_geometric_align: true`
- 需同时开启 `two_stage.enable_temporal_concat: true`
- 可选提供 `two_stage.transform_matrices_path`
  - 文件格式：`float32` 二进制，shape=`[adj_num, 9]`
  - 每个 3x3 变换矩阵按 row-major 存储
  - 不提供时默认使用单位矩阵（等效无几何位移）
- 或提供 `two_stage.transform_sequence_dir`
  - 目录下每个 `*.bin` 代表一帧变换，按文件名排序循环使用
  - 每个文件大小应为 `adj_num * 9 * sizeof(float)`

可根据样本 YAML 批量生成按帧变换序列：

```bash
python tools/generate_transform_sequence_from_samples.py \
  --model-cfg /data/bevdet-tensorrt-cpp/cfgs/bevdet_lt_depth.yaml \
  --samples-dir /data/bevdet-tensorrt-cpp/sample0 \
  --adj-num 8 \
  --out-dir sample0/bevpool_indices/transform_seq_from_samples
```

可用以下脚本生成单位矩阵文件：

```bash
python tools/generate_identity_transforms.py \
  --adj-num 8 \
  --out sample0/bevpool_indices/identity_transforms.bin
```

可选：打印阶段级耗时（two-stage）

- `runtime.print_stage_timing: true`

`tio_benchmark` 会额外输出：

- `stage_mean(ms): img=..., bevpool=..., align=..., bev=..., total=...`

## 1) Generate Calibration Batches

```bash
python tools/generate_dummy_calib.py --out-dir sample0 --count 8 --shape 1,6,3,256,704
```

## 2) Build Engine by trtexec (optional baseline)

```bash
python tools/export_engine.py \
  --onnx model/bevdet.onnx \
  --engine engine/bevdet_fp16.engine \
  --fp16 \
  --min-shapes input:1x6x3x256x704 \
  --opt-shapes input:4x6x3x256x704 \
  --max-shapes input:8x6x3x256x704
```

说明：

- `tools/export_engine.py` 已兼容 TRT8/TRT10 workspace 参数差异：
  - TRT8 常见 `--workspace=4096`
  - TRT10 常见 `--memPoolSize=workspace:4096`
- 脚本会自动探测 `trtexec --help` 并选择参数。

INT8 vs FP16 快速报告（性能 + 输出误差）：

```bash
cmake -S . -B build -DTIO_ENABLE_BENCHMARK=ON
cmake --build build -j --target tio_compare_engines

python tools/int8_fp16_report.py \
  --fp16-engine engine/bevdet_fp16.engine \
  --int8-engine engine/bevdet_int8.engine \
  --compare-bin build/tio_compare_engines \
  --json-out reports/int8_fp16_report.json
```

## 3) Build C++ Runtime

默认 CUDA / TensorRT 路径写在 `tools/build.sh` 顶部（`TIO_DEFAULT_*`），一般只需：

```bash
# Local（如 GTX 1080，sm_61）
tools/build.sh -p local

# RTX 4090（sm_89，并打开 benchmark）
tools/build.sh -p 4090

# 需要时：清目录重编、改路径
tools/build.sh -p local -c
tools/build.sh -p local -u /usr/local/cuda -t /usr
```

## 4) Run Demo

默认配置文件写在 `tools/run.sh` 顶部（`TIO_DEFAULT_CONFIG`，相对仓库根）：

```bash
tools/run.sh -m demo
# 或指定配置
tools/run.sh -m demo -c cfgs/default.yaml
```

## 5) Run Benchmark

```bash
tools/run.sh -m benchmark
tools/run.sh -m benchmark -c configure.yaml
```

## 6) Profile

```bash
tools/run.sh -m demo -n
tools/run.sh -m demo -n -c configure.yaml
```

## 7) Plugin Correctness Check

```bash
python tools/plugin_correctness.py --input sample0/input.npy --output sample0/output.npy
```

`BevPoolPlugin` 当前输入约定（one-branch 迁移中的最小协议）：

- Input-0: `bev_template` (`float`, shape 用于输出维度)
- Input-1: `depth` (`float`)
- Input-2: `feat` (`float`)
- Input-3~7: `ranks_depth / ranks_feat / ranks_bev / interval_starts / interval_lengths` (`int32`)
- Output-0: `BEV_feat` (`float`)

插件内核已从 identity 占位升级为真实 `bev_pool_v2` 聚合逻辑。

可通过独立目标做内核数值单测：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_bevpool_kernel_test
./build_plugin/tio_bevpool_kernel_test
```

Align kernel 单测：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_alignbev_kernel_test
./build_plugin/tio_alignbev_kernel_test
```

Gather kernel 单测：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_gatherbev_kernel_test
./build_plugin/tio_gatherbev_kernel_test
```

`AlignBevPlugin` 当前输入约定（最小协议）：

- Input-0: `adj_feat` (`float`, shape `[adj_num, C, H, W]`)
- Input-1: `transform` (`float`, shape `[adj_num, 9]`, 每个 3x3 row-major)
- Output-0: `aligned_adj_feat` (`float`, shape 同 Input-0)

`GatherBevPlugin` 当前输入约定（最小协议）：

- Input-0: `adj_feat` (`float`, shape `[B, adj_num, C, H, W]`)
- Input-1: `curr_feat` (`float`, shape `[B, C, H, W]`)
- Input-2: `flag` (`int32`, shape `[B]`; `0` 表示回退为当前帧复制，`1` 表示拼接邻帧)
- Output-0: `bev_queue` (`float`, shape `[B, (adj_num+1)*C, H, W]`)

插件 `enqueue` 冒烟测试（覆盖插件外壳 + CUDA 内核）：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_plugin_enqueue_smoke_test
./build_plugin/tio_plugin_enqueue_smoke_test
```

最小 TRT 建图/序列化/反序列化/执行集成测试（同一网络挂载 `BevPool+Align+Gather`）：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_plugin_minimal_trt_test
./build_plugin/tio_plugin_minimal_trt_test
```

`Gather -> Align` 串联子图集成测试（验证插件间真实张量衔接）：

```bash
cmake -S . -B build_plugin -DTIO_ENABLE_PLUGIN=ON
cmake --build build_plugin -j --target tio_plugin_chain_trt_test
./build_plugin/tio_plugin_chain_trt_test
```

## 8) Known Build/Runtime Pitfalls (from real runs)

1. **TensorRT not found at CMake configure**

   - Symptom: `TensorRT not found. Set TensorRT_ROOT or TENSORRT_ROOT env.`
   - Check: headers `NvInfer.h`, `NvOnnxParser.h`; libs `libnvinfer.so`, `libnvonnxparser.so`.
   - Fix: install TensorRT dev packages and pass `-t /usr` if needed.

2. **`trtexec` not found after package install**

   - Symptom: `trtexec: command not found`
   - Check: `/usr/src/tensorrt/bin/trtexec`
   - Fix:

     ```bash
     sudo ln -sf /usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
     ```

3. **Driver/runtime mismatch (`+cuda13.x` packages on CUDA 12.4 host)**

   - Symptom: `CUDA driver version is insufficient for CUDA runtime version`
   - Fix: switch TensorRT packages to CUDA 12.4 variant (for example `10.1.0.27-1+cuda12.4`).

4. **TRT10 CLI incompatible with legacy `--workspace`**

   - Symptom: `Unknown option: --workspace ...`
   - Fix: use `--memPoolSize=workspace:<MiB>` or run updated `tools/export_engine.py`.

## Known Gaps

- Calibration reader expects flat float32 `.bin` batches with exact tensor bytes.
- Full BEVDet parser wiring requires model-specific tensor names in config.

## Script Organization (current)

- Build:
  - `tools/build.sh`
- Run:
  - `tools/run.sh`
  - `tools/run_demo.sh` (demo alias)
  - `tools/profile.sh` (demo+nsys alias)
- Export/Benchmark:
  - `tools/export_engine.py`
  - `tools/run_checkpoint_trt.sh`
- Quantization report:
  - `tools/int8_fp16_report.py`
  - `tools/run_ptq_baseline.sh`
- Data prep:
  - `tools/generate_dummy_calib.py`
  - `tools/generate_bevpool_indices.py`
  - `tools/generate_transform_sequence_from_samples.py`
  - `tools/generate_identity_transforms.py`
