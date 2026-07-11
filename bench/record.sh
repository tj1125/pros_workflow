#!/usr/bin/env bash
# One-shot full-run capture: start whole-machine monitoring, run your task, press Enter
# to stop and render the figure. Captures the ENTIRE execution (GPU/CPU/RAM/power) —
# system-wide, so it covers the workflow, A2A servers and Ollama together.
#
#   bash bench/record.sh <name> ["chart title"]
#   e.g.  bash bench/record.sh grasp_full "Full grasp task — AMD Strix Halo"
#
# Writes bench/results/<name>.csv / .summary.json / .png
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NAME="${1:?usage: bash bench/record.sh <name> [\"chart title\"]}"
TITLE="${2:-Full task run — AMD Strix Halo (Radeon 8060S)}"
INTERVAL="${INTERVAL:-1}"
mkdir -p bench/results
OUT="bench/results/$NAME"

python bench/monitor_resources.py -o "$OUT" --interval "$INTERVAL" --label "$NAME" &
MON=$!
trap 'kill "$MON" 2>/dev/null || true' EXIT

cat <<EOF

>>> Monitoring the whole machine (PID $MON). Start your full task now.
>>> Press Enter the moment the task finishes — this stops recording and plots.
EOF
read -r _

kill "$MON" 2>/dev/null || true
wait "$MON" 2>/dev/null || true
trap - EXIT

python bench/plot_resources.py "$OUT.csv" --title "$TITLE"
echo ">>> done: $OUT.csv  |  $OUT.summary.json  |  $OUT.png"
