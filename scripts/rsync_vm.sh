#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "push" && "$MODE" != "pull" ]]; then
  echo "usage: $0 <push|pull>" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

INSTANCE="${INSTANCE:-bpe-trainer-64}"
ZONE="${ZONE:-us-west1-b}"
REMOTE_USER="${REMOTE_USER:-$USER}"
REMOTE_DIR="${REMOTE_DIR:-/home/$REMOTE_USER/assignment1-basics}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/google_compute_engine}"
DRY_RUN="${DRY_RUN:-0}"
DELETE_FLAG="${DELETE_FLAG:-0}"

HOST="$(
  gcloud compute instances describe "$INSTANCE" \
    --zone="$ZONE" \
    --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
)"

SSH_OPTS=(
  -i "$SSH_KEY"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="$HOME/.ssh/google_compute_known_hosts"
)

RSYNC_OPTS=(
  -avz
  --progress
  --exclude-from="$REPO_ROOT/.rsyncignore"
)

if [[ "$DRY_RUN" == "1" ]]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [[ "$DELETE_FLAG" == "1" ]]; then
  RSYNC_OPTS+=(--delete)
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$HOST" "mkdir -p '$REMOTE_DIR'"

if [[ "$MODE" == "push" ]]; then
  rsync "${RSYNC_OPTS[@]}" -e "ssh ${SSH_OPTS[*]}" \
    "$REPO_ROOT/" "$REMOTE_USER@$HOST:$REMOTE_DIR/"
else
  rsync "${RSYNC_OPTS[@]}" -e "ssh ${SSH_OPTS[*]}" \
    "$REMOTE_USER@$HOST:$REMOTE_DIR/" "$REPO_ROOT/"
fi
