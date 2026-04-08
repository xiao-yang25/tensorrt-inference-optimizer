#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configure.yaml}"

mkdir -p build
cmake -S . -B build
cmake --build build -j
./build/tio_demo "${CONFIG}"
