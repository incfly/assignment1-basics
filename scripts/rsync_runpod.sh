#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "push" && "$MODE" != "pull" ]]; then
  echo "usage: $0 <push|pull>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/workspace/assignment1-basics}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_FLAG="${DELETE_FLAG:-0}"

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

RSYNC_OPTS=(
  -avz
  --progress
  --exclude-from="$REPO_ROOT/.rsyncignore"
  --exclude="/runs/"
  --exclude="/checkpoints/"
)
if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi
if [[ "$DELETE_FLAG" == "1" ]]; then
  RSYNC_OPTS+=(--delete)
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$RUNPOD_HOST" \
  "mkdir -p '$REMOTE_DIR' && (command -v rsync >/dev/null || (apt-get update && apt-get install -y rsync))"

if [[ "$MODE" == "push" ]]; then
  rsync "${RSYNC_OPTS[@]}" -e "ssh ${SSH_OPTS[*]}" "$REPO_ROOT/" "$REMOTE_USER@$RUNPOD_HOST:$REMOTE_DIR/"
else
  rsync "${RSYNC_OPTS[@]}" -e "ssh ${SSH_OPTS[*]}" "$REMOTE_USER@$RUNPOD_HOST:$REMOTE_DIR/" "$REPO_ROOT/"
fi
