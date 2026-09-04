#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/eval_finetuned.yaml}"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[eval-finetuned] Running finetuned model evaluation with config: $CONFIG"
uv run python -m src.cli.evaluate --config "$CONFIG"
echo "[eval-finetuned] Done."
