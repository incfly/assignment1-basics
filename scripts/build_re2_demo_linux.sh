#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RE2_PREFIX="${ROOT_DIR}/.local/re2"
BUILD_DIR="${ROOT_DIR}/re2_demo/build"
DEFAULT_PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3"
if [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

command -v cmake >/dev/null || { echo "cmake is required"; exit 1; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "${PYTHON_BIN} is required"; exit 1; }

if [[ ! -f "${RE2_PREFIX}/lib/cmake/re2/re2Config.cmake" ]]; then
  echo "RE2 is not installed in ${RE2_PREFIX}. Run scripts/bootstrap_re2_linux.sh first."
  exit 1
fi

cmake -S "${ROOT_DIR}/re2_demo" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${RE2_PREFIX}" \
  -DPython3_EXECUTABLE="$(command -v "${PYTHON_BIN}")"

cmake --build "${BUILD_DIR}" --parallel

echo "Built Python extension in ${ROOT_DIR}/re2_demo"
