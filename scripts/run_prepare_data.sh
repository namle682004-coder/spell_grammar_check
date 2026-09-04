#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/train_finetune.yaml}"

if [ -f .env ]; then
  echo "[prepare] Loading .env..."
  set -a; source .env; set +a
fi

echo "[prepare] Running data preparation with config: $CONFIG"
uv run python -m src.cli.prepare_data --config "$CONFIG"
echo "[prepare] Done."
