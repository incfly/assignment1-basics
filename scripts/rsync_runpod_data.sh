#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "push" && "$MODE" != "pull" ]]; then
  echo "usage: $0 <push|pull>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-$REPO_ROOT/../cs336-data/tinystory}"
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-/workspace/cs336-data/tinystory}"

PYTHON_CMD="${PYTHON_CMD:-uv run python}"
read -r REMOTE_USER RUNPOD_HOST RUNPOD_PORT RUNPOD_KEY_FROM_RESOLVER < <($PYTHON_CMD "$SCRIPT_DIR/runpod_resolve_ssh.py")
if [[ -z "${RUNPOD_KEY:-}" && "${RUNPOD_KEY_FROM_RESOLVER:-}" != "-" ]]; then
  RUNPOD_KEY="$RUNPOD_KEY_FROM_RESOLVER"
fi
if [[ -n "${RUNPOD_KEY:-}" ]]; then
  RUNPOD_KEY="${RUNPOD_KEY/#\~/$HOME}"
fi

SSH_OPTS=(
  -p "$RUNPOD_PORT"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/runpod_known_hosts"
)
if [[ -n "${RUNPOD_KEY:-}" ]]; then
  SSH_OPTS+=(-i "$RUNPOD_KEY")
fi

FILES=(
  "TinyStories-train.txt-tokenized.bin"
  "TinyStories-valid.txt-tokenized.bin"
)

if [[ "$MODE" == "push" ]]; then
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$RUNPOD_HOST" \
    "mkdir -p '$REMOTE_DATA_DIR' && (command -v rsync >/dev/null || (apt-get update && apt-get install -y rsync))"
  for file in "${FILES[@]}"; do
    rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" "$LOCAL_DATA_DIR/$file" "$REMOTE_USER@$RUNPOD_HOST:$REMOTE_DATA_DIR/"
  done
else
  mkdir -p "$LOCAL_DATA_DIR"
  for file in "${FILES[@]}"; do
    rsync -avz --progress -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$RUNPOD_HOST:$REMOTE_DATA_DIR/$file" "$LOCAL_DATA_DIR/"
  done
fi
