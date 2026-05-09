#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../../cs336-data}"

mkdir -p "$DATA_ROOT"

fetch() {
  local dest="$1"
  local url="$2"
  if [[ -f "$dest" ]]; then
    echo "exists: $dest"
    return
  fi
  echo "download: $dest"
  curl -L "$url" -o "$dest"
}

fetch "$DATA_ROOT/TinyStories-train.txt" \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
fetch "$DATA_ROOT/TinyStories-valid.txt" \
  "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

fetch "$DATA_ROOT/owt_train.txt.gz" \
  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"
if [[ ! -f "$DATA_ROOT/owt_train.txt" ]]; then
  gunzip -k "$DATA_ROOT/owt_train.txt.gz"
fi

fetch "$DATA_ROOT/owt_valid.txt.gz" \
  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz"
if [[ ! -f "$DATA_ROOT/owt_valid.txt" ]]; then
  gunzip -k "$DATA_ROOT/owt_valid.txt.gz"
fi

if [[ ! -f "$DATA_ROOT/tiny-1000.txt" ]]; then
  head -n 1000 "$DATA_ROOT/TinyStories-train.txt" > "$DATA_ROOT/tiny-1000.txt"
fi

echo "data root: $DATA_ROOT"
