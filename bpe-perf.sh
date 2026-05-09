#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../cs336-data}"

# On macOS, use util-linux setsid so we can sample the whole process group.
SETSID_BIN="$(brew --prefix util-linux)/bin/setsid"
PID_FILE="$(mktemp /tmp/bpe-session.XXXXXX.pid)"
OUTPUT_CSV="${OUTPUT_CSV:-usage.csv}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"
INPUT_FILE="${INPUT_FILE:-$DATA_ROOT/TinyStories-train.txt}"
VOCAB_SIZE="${VOCAB_SIZE:-500}"
PRETOKEN_WORKER="${PRETOKEN_WORKER:-8}"
PRETOKEN_CHUNK="${PRETOKEN_CHUNK:-8}"
RUN_LOG="${RUN_LOG:-/tmp/bpe-perf-run.log}"

cleanup() {
  rm -f "$PID_FILE"
}
trap cleanup EXIT

"$SETSID_BIN" sh -c '
  echo $$ > "$1"
  cd "$6"
  exec uv run python cs336_basics/bpe_merge.py "$2" \
    --vocab-size "$3" --pretoken-worker "$4" --pretoken-chunk "$5"
' sh "$PID_FILE" "$INPUT_FILE" "$VOCAB_SIZE" "$PRETOKEN_WORKER" "$PRETOKEN_CHUNK" "$SCRIPT_DIR" \
  >"$RUN_LOG" 2>&1 &
LAUNCHER_PID=$!

# Quirky thing about mac setsid util-linux. Not behave exactly as Linux.
for _ in $(seq 1 50); do
  if [[ -s "$PID_FILE" ]]; then
    break
  fi
  sleep 0.1
done

if [[ ! -s "$PID_FILE" ]]; then
  echo "failed to capture session leader pid" >&2
  exit 1
fi

SESSION_PID="$(<"$PID_FILE")"
PGID="$(ps -o pgid= -p "$SESSION_PID" | tr -d ' ')"

echo "t_s,cpu_pct,rss_kb,proc_count" > "$OUTPUT_CSV"
START_TS="$(perl -MTime::HiRes=time -e 'printf "%.3f", time')"

while true; do
  NOW_TS="$(perl -MTime::HiRes=time -e 'printf "%.3f", time')"
  ELAPSED="$(perl -e "printf \"%.3f\", $NOW_TS - $START_TS")"
  SAMPLE="$(ps -g "$PGID" -o pid=,%cpu=,rss= 2>/dev/null || true)"
  if [[ -z "${SAMPLE//[[:space:]]/}" ]]; then
    break
  fi

  awk -v t="$ELAPSED" '
    NF >= 3 {
      count += 1
      cpu += $2
      rss += $3
    }
    END { printf "%s,%.2f,%d,%d\n", t, cpu, rss, count }
  ' <<<"$SAMPLE" >> "$OUTPUT_CSV"

  sleep "$SAMPLE_INTERVAL"
done

wait "$LAUNCHER_PID"
