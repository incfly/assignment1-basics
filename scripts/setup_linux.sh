#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DATA="${INSTALL_DATA:-0}"
DATA_ROOT="${DATA_ROOT:-${ROOT_DIR}/../cs336-data}"
SMOKE_DATA="${SMOKE_DATA:-${DATA_ROOT}/tiny-1000.txt}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup_linux.sh is intended for Linux hosts."
  exit 1
fi

install_apt_deps() {
  if ! command -v apt-get >/dev/null; then
    echo "apt-get not found. Install these packages manually:"
    echo "  tmux rsync build-essential cmake git pkg-config python3-dev libabsl-dev curl"
    return
  fi

  sudo apt-get update
  sudo apt-get install -y \
    tmux \
    rsync \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3-dev \
    libabsl-dev \
    curl
}

install_uv() {
  if command -v uv >/dev/null; then
    return
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
}

cd "${ROOT_DIR}"
install_apt_deps
install_uv

export PATH="${HOME}/.local/bin:${PATH}"

uv sync
./scripts/bootstrap_re2_linux.sh
./scripts/build_re_cpp_linux.sh

if [[ "${INSTALL_DATA}" == "1" ]]; then
  ./scripts/init_data.sh
fi

if [[ -f "${SMOKE_DATA}" ]]; then
  uv run python -m cs336_basics.bpe.merge "${SMOKE_DATA}" --vocab-size 300
else
  echo "skip smoke test: ${SMOKE_DATA} not found"
  echo "set DATA_ROOT or run INSTALL_DATA=1 ./scripts/setup_linux.sh to initialize data"
fi

echo "Linux setup complete."
