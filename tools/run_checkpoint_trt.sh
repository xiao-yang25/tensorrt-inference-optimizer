#!/usr/bin/env bash
set -euo pipefail

# Build + benchmark the two-stage ONNX artifacts shipped under checkpoint/.
# This matches the export layout described in:
# https://github.com/LCH1238/BEVDet/blob/export/README_zh-CN.md

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

IMG_ONNX="${1:-checkpoint/img_stage_lt_d.onnx}"
BEV_ONNX="${2:-checkpoint/bev_stage_lt_d.onnx}"

mkdir -p engine

IMG_ENGINE="engine/$(basename "${IMG_ONNX}" .onnx)_fp16.engine"
BEV_ENGINE="engine/$(basename "${BEV_ONNX}" .onnx)_fp16.engine"

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found in PATH. On many Linux installs it lives at:" >&2
  echo "  /usr/src/tensorrt/bin/trtexec" >&2
  exit 1
fi

echo "[1/2] Build + benchmark img stage: ${IMG_ONNX}"
trtexec \
  --onnx="${IMG_ONNX}" \
  --saveEngine="${IMG_ENGINE}" \
  --fp16 \
  --memPoolSize=workspace:4096

echo
echo "[2/2] Build + benchmark bev stage: ${BEV_ONNX}"
trtexec \
  --onnx="${BEV_ONNX}" \
  --saveEngine="${BEV_ENGINE}" \
  --fp16 \
  --memPoolSize=workspace:4096

echo
echo "Done."
echo "Engines:"
ls -lh "${IMG_ENGINE}" "${BEV_ENGINE}"
