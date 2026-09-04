#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-vietnamese-grammar-qwen1.5b-shynbui-lora}"
SOURCE="${2:-}"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

ARGS=(--run-name "$RUN_NAME")
if [ -n "$SOURCE" ]; then
  ARGS+=(--source "$SOURCE")
fi

echo "[promote] Run name: $RUN_NAME"
uv run python -m src.cli.promote "${ARGS[@]}"
echo "[promote] API will load outputs/production/current (or set FINETUNE_MODEL in .env)."
