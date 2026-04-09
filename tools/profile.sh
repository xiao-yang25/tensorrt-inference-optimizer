#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ge 1 ]]; then
  exec "${ROOT_DIR}/tools/run.sh" -m demo -n -c "$1"
fi
exec "${ROOT_DIR}/tools/run.sh" -m demo -n
