#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build_ci"
REPORT="${ROOT_DIR}/docs/verification_report.md"
LOG_DIR="${ROOT_DIR}/reports/ci"
LOG_FILE="${LOG_DIR}/oneclick_ci.log"

mkdir -p "${LOG_DIR}" "$(dirname "${REPORT}")"
: > "${LOG_FILE}"

PASS=0
FAIL=0

run_step() {
  local name="$1"
  shift
  echo "==> ${name}" | tee -a "${LOG_FILE}"
  if "$@" >> "${LOG_FILE}" 2>&1; then
    echo "| ${name} | PASS |" >> "${REPORT}.table"
    PASS=$((PASS + 1))
  else
    echo "| ${name} | FAIL |" >> "${REPORT}.table"
    FAIL=$((FAIL + 1))
  fi
}

echo "| Step | Result |" > "${REPORT}.table"
echo "|---|---|" >> "${REPORT}.table"

run_step "CMake Configure (plugin+benchmark+tests)" \
  cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DTIO_ENABLE_PLUGIN=ON -DTIO_ENABLE_BENCHMARK=ON -DTIO_ENABLE_TESTS=ON

run_step "Build All Targets" \
  cmake --build "${BUILD_DIR}" -j

run_step "CTest (plugin test suite)" \
  ctest --test-dir "${BUILD_DIR}" --output-on-failure

{
  echo "# Verification Report"
  echo
  echo "- Script: \`tools/oneclick_ci.sh\`"
  echo "- Build dir: \`build_ci\`"
  echo "- Log: \`reports/ci/oneclick_ci.log\`"
  echo
  echo "## Summary"
  echo
  echo "- PASS: ${PASS}"
  echo "- FAIL: ${FAIL}"
  echo
  echo "## Step Results"
  echo
  cat "${REPORT}.table"
  echo
  echo "## Conclusion"
  echo
  if [[ "${FAIL}" -eq 0 ]]; then
    echo "All one-click compile/test steps passed."
  else
    echo "Some steps failed. Check \`reports/ci/oneclick_ci.log\` for details."
  fi
} > "${REPORT}"

rm -f "${REPORT}.table"

if [[ "${FAIL}" -ne 0 ]]; then
  echo "oneclick_ci failed, see ${LOG_FILE}" >&2
  exit 1
fi

echo "oneclick_ci passed. Report: ${REPORT}"
