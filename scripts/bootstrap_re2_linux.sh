#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
RE2_SRC_DIR="${THIRD_PARTY_DIR}/re2"
RE2_BUILD_DIR="${RE2_SRC_DIR}/build"
RE2_INSTALL_DIR="${ROOT_DIR}/.local/re2"
RE2_TAG="${RE2_TAG:-2025-11-05}"

command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v cmake >/dev/null || { echo "cmake is required"; exit 1; }
command -v pkg-config >/dev/null || { echo "pkg-config is required"; exit 1; }

if ! pkg-config --exists absl_base; then
  cat <<'EOF'
Missing Abseil development files.
On Ubuntu/Debian, run:
  sudo apt-get update
  sudo apt-get install -y build-essential cmake git pkg-config python3-dev libabsl-dev
EOF
  exit 1
fi

mkdir -p "${THIRD_PARTY_DIR}"

if [[ ! -d "${RE2_SRC_DIR}/.git" ]]; then
  git clone --depth 1 --branch "${RE2_TAG}" https://github.com/google/re2 "${RE2_SRC_DIR}"
else
  git -C "${RE2_SRC_DIR}" fetch --depth 1 origin "refs/tags/${RE2_TAG}:refs/tags/${RE2_TAG}"
  git -C "${RE2_SRC_DIR}" checkout "${RE2_TAG}"
fi

cmake -S "${RE2_SRC_DIR}" -B "${RE2_BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${RE2_INSTALL_DIR}" \
  -DRE2_TEST=OFF \
  -DRE2_BENCHMARK=OFF

cmake --build "${RE2_BUILD_DIR}" --parallel
cmake --install "${RE2_BUILD_DIR}"

echo "RE2 installed to ${RE2_INSTALL_DIR}"
