#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# 默认路径（本机按需改这里即可，一般不必再传 -c / -b）
# ---------------------------------------------------------------------------
: "${TIO_DEFAULT_CONFIG:=configure.yaml}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
MODE="demo"
CONFIG="${ROOT_DIR}/${TIO_DEFAULT_CONFIG}"
NSYS="0"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra args to binary]

短参数:
  -m <demo|benchmark>   运行目标（默认 demo）
  -c <yaml>             配置文件（相对项目根或绝对路径；默认见脚本 TIO_DEFAULT_CONFIG）
  -b <dir>              CMake 构建目录（默认 <repo>/build）
  -n                    用 nsys profile --stats=true 包裹
  -h                    帮助
  --                    之后参数原样传给可执行文件

示例:
  $(basename "$0")
  $(basename "$0") -m benchmark
  $(basename "$0") -c cfgs/default.yaml
  $(basename "$0") -m demo -n
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m)
      MODE="$2"
      shift 2
      ;;
    -c)
      if [[ "$2" == /* ]]; then
        CONFIG="$2"
      else
        CONFIG="${ROOT_DIR}/$2"
      fi
      shift 2
      ;;
    -b)
      BUILD_DIR="$2"
      shift 2
      ;;
    -n)
      NSYS="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${MODE}" in
  demo) BIN="${BUILD_DIR}/tio_demo" ;;
  benchmark) BIN="${BUILD_DIR}/tio_benchmark" ;;
  *)
    echo "Invalid -m mode: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ ! -x "${BIN}" ]]; then
  cat >&2 <<EOF
Binary not found: ${BIN}
请先编译，例如:
  scripts/build.sh -p local
  scripts/build.sh -p 4090
EOF
  exit 1
fi

if [[ "${NSYS}" == "1" ]]; then
  exec nsys profile --stats=true "${BIN}" "${CONFIG}" "${EXTRA_ARGS[@]}"
fi

exec "${BIN}" "${CONFIG}" "${EXTRA_ARGS[@]}"
