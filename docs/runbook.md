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

```bash
cmake -S . -B build
cmake --build build -j
```

## 4) Run Demo

```bash
./build/tio_demo configure.yaml
```

## 5) Run Benchmark

```bash
./build/tio_benchmark configure.yaml
```

## 6) Profile

```bash
nsys profile --stats=true ./build/tio_demo configure.yaml
```

## 7) Plugin Correctness Check

```bash
python tools/plugin_correctness.py --input sample0/input.npy --output sample0/output.npy
```

## Known Gaps

- BEVPool plugin kernel is currently an identity placeholder for integration scaffolding.
- Calibration reader expects flat float32 `.bin` batches with exact tensor bytes.
- Full BEVDet parser wiring requires model-specific tensor names in config.
