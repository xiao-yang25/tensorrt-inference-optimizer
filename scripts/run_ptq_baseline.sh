#!/usr/bin/env bash
set -euo pipefail

# One-shot PTQ baseline:
# 1) build FP16 engine
# 2) build INT8 engine (reuse/create calib cache)
# 3) run perf+diff report

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

ONNX_PATH="${1:-model/bevdet.onnx}"
INPUT_NAME="${2:-input}"

FP16_ENGINE="${ROOT_DIR}/output/engines/bevdet_fp16.engine"
INT8_ENGINE="${ROOT_DIR}/output/engines/bevdet_int8.engine"
CALIB_CACHE="${ROOT_DIR}/output/cache/calib.cache"
REPORT_JSON="${ROOT_DIR}/output/reports/int8_fp16_report.json"
BUILD_DIR="${ROOT_DIR}/build"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX not found: ${ONNX_PATH}" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/output/engines" "${ROOT_DIR}/output/cache" "${ROOT_DIR}/output/reports"

echo "[1/5] Build compare binary"
cmake -S . -B "${BUILD_DIR}" -DTIO_ENABLE_BENCHMARK=ON
cmake --build "${BUILD_DIR}" -j --target tio_compare_engines

echo "[2/5] Build FP16 engine"
python scripts/export_engine.py \
  --onnx "${ONNX_PATH}" \
  --engine "${FP16_ENGINE}" \
  --fp16 \
  --min-shapes "${INPUT_NAME}:1x6x3x256x704" \
  --opt-shapes "${INPUT_NAME}:1x6x3x256x704" \
  --max-shapes "${INPUT_NAME}:1x6x3x256x704"

echo "[3/5] Ensure calibration batches"
if [[ ! -f sample0/calib_batch_000.bin ]]; then
  python scripts/generate_dummy_calib.py --out-dir sample0 --count 8 --shape 1,6,3,256,704
fi

echo "[4/5] Build INT8 engine"
python scripts/export_engine.py \
  --onnx "${ONNX_PATH}" \
  --engine "${INT8_ENGINE}" \
  --int8 \
  --calib "${CALIB_CACHE}" \
  --min-shapes "${INPUT_NAME}:1x6x3x256x704" \
  --opt-shapes "${INPUT_NAME}:1x6x3x256x704" \
  --max-shapes "${INPUT_NAME}:1x6x3x256x704"

echo "[5/5] Generate FP16 vs INT8 report"
python scripts/int8_fp16_report.py \
  --fp16-engine "${FP16_ENGINE}" \
  --int8-engine "${INT8_ENGINE}" \
  --compare-bin "${BUILD_DIR}/tio_compare_engines" \
  --json-out "${REPORT_JSON}"

echo "Done: ${REPORT_JSON}"
