#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup_train_linux.sh is intended for Linux hosts."
  exit 1
fi

if [[ "$(id -u)" == "0" ]]; then
  SUDO=""
else
  SUDO="sudo"
fi

if command -v apt-get >/dev/null; then
  $SUDO apt-get update
  $SUDO apt-get install -y tmux rsync git curl ca-certificates
else
  echo "apt-get not found; skipping package install"
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

cd "$ROOT_DIR"
TRAIN_SETUP_MODE="${TRAIN_SETUP_MODE:-system-torch}"
if [[ "$TRAIN_SETUP_MODE" == "uv" ]]; then
  uv sync
  PYTHON_RUN=(uv run python)
else
  python3 -m pip install --no-cache-dir einops einx jaxtyping tqdm
  PYTHON_RUN=(python3)
fi

"${PYTHON_RUN[@]}" - <<'PY'
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device", torch.cuda.get_device_name(0))
PY

echo "Training setup complete."
