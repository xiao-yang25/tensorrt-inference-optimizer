# Runbook

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

## Known Gaps

- BEVPool plugin kernel is currently an identity placeholder for integration scaffolding.
- Calibration reader expects flat float32 `.bin` batches with exact tensor bytes.
- Full BEVDet parser wiring requires model-specific tensor names in config.
