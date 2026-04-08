#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configure.yaml}"

nsys profile --stats=true ./build/tio_demo "${CONFIG}"
