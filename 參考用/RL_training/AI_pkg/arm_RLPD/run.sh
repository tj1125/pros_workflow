#!/usr/bin/env bash
# Helper launcher for training/collection.
# Examples:
#   ./run.sh offline --data_path data/offline_buffer.jsonl --batch_size 4096 --device mps
#   ./run.sh hybrid  --offline_path data/offline_buffer.jsonl --batch_size 2048 --device mps
#   ./run.sh online  --device mps
#   ./run.sh collect --out data/offline_buffer.jsonl

set -euo pipefail

export PYTHONPATH=$(pwd):${PYTHONPATH:-}

cmd=${1:-offline}
shift || true

case "$cmd" in
  offline)
    # Defaults are handled inside the script (batch_size=256, device=mps if available)
    python -m src.train.train_offline "$@" ;;
  online)
    python -m src.train.train_online "$@" ;;
  hybrid)
    # Defaults are handled inside the script (batch_size=256, device=mps if available)
    python -m src.train.train_hybrid "$@" ;;
  collect)
    python -m src.hf.collect_cli "$@" ;;
  infer)
    python -m src.inference.run_policy "$@" ;;
  *)
    echo "Usage: $0 [offline|online|hybrid|collect|infer] [--args...]" >&2
    exit 1 ;;
esac
