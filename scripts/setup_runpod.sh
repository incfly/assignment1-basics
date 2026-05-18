#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/workspace/assignment1-basics}"
PYTHON_CMD="${PYTHON_CMD:-uv run python}"

"$SCRIPT_DIR/rsync_runpod.sh" push
"$SCRIPT_DIR/rsync_runpod_data.sh" push

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

ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$RUNPOD_HOST" "cd '$REMOTE_DIR' && ./scripts/setup_train_linux.sh"
