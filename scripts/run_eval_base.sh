#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/eval_base.yaml}"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[eval-base] Running base model evaluation with config: $CONFIG"
uv run python -m src.cli.evaluate --config "$CONFIG"
echo "[eval-base] Done."
