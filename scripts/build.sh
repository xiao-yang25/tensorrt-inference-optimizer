#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认路径（本机按需改这里即可，一般不必再传 -u / -t）
# ---------------------------------------------------------------------------
: "${TIO_DEFAULT_CUDA_ROOT:=/usr/local/cuda-12.4}"
: "${TIO_DEFAULT_TENSORRT_ROOT:=/usr}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BUILD_TYPE="Release"
CUDA_ROOT="${CUDA_HOME:-${TIO_DEFAULT_CUDA_ROOT}}"
CUDA_ARCH=""
TENSORRT_ROOT="${TIO_DEFAULT_TENSORRT_ROOT}"
ENABLE_PLUGIN="OFF"
ENABLE_BENCHMARK="OFF"
STRICT_TARGETS="OFF"
CLEAN="0"
PRESET=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

短参数（路径默认见脚本顶部 TIO_DEFAULT_*，需要时再 -u / -t）:
  -p <local|4090>   预设：local=sm61；4090=sm89 并打开 benchmark
  -c                编译前清空 build/
  -t <path>         TensorRT 根目录（覆盖默认）
  -u <path>         CUDA Toolkit 根目录（覆盖默认，仍优先 \$CUDA_HOME）
  -a <arch>         CMAKE_CUDA_ARCHITECTURES，如 61 / 89 / 86;89
  -P                打开 plugin 目标
  -B                打开 benchmark 目标
  -S                可选目标缺失时直接报错（严格模式）
  -h                帮助

示例:
  $(basename "$0") -p local
  $(basename "$0") -p 4090 -c
  $(basename "$0") -a 89 -B -P
EOF
}

while getopts "p:ct:u:a:PBSh" opt; do
  case "${opt}" in
    p) PRESET="${OPTARG}" ;;
    c) CLEAN="1" ;;
    t) TENSORRT_ROOT="${OPTARG}" ;;
    u) CUDA_ROOT="${OPTARG}" ;;
    a) CUDA_ARCH="${OPTARG}" ;;
    P) ENABLE_PLUGIN="ON" ;;
    B) ENABLE_BENCHMARK="ON" ;;
    S) STRICT_TARGETS="ON" ;;
    h) usage; exit 0 ;;
    *) usage; exit 1 ;;
  esac
done

case "${PRESET}" in
  "")
    ;;
  local)
    CUDA_ARCH="${CUDA_ARCH:-61}"
    ;;
  4090)
    CUDA_ARCH="${CUDA_ARCH:-89}"
    ENABLE_BENCHMARK="ON"
    ;;
  *)
    echo "Invalid -p preset: ${PRESET} (use local or 4090)" >&2
    usage
    exit 1
    ;;
esac

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
  -S "${ROOT_DIR}"
  -B "${BUILD_DIR}"
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
  -DCUDAToolkit_ROOT="${CUDA_ROOT}"
  -DCMAKE_CUDA_COMPILER="${CUDA_ROOT}/bin/nvcc"
  -DTensorRT_ROOT="${TENSORRT_ROOT}"
  -DTIO_ENABLE_PLUGIN="${ENABLE_PLUGIN}"
  -DTIO_ENABLE_BENCHMARK="${ENABLE_BENCHMARK}"
  -DTIO_STRICT_TARGETS="${STRICT_TARGETS}"
)

if [[ -n "${CUDA_ARCH}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
fi

echo "[build] configure: cmake ${CMAKE_ARGS[*]}"
cmake "${CMAKE_ARGS[@]}"

echo "[build] compile"
cmake --build "${BUILD_DIR}" -j

echo "[build] done"
