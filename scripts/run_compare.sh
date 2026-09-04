#!/usr/bin/env bash
set -euo pipefail

FINETUNED_RUN_NAME="${1:-vietnamese-grammar-qwen1.5b-shynbui-lora}"
SPLIT="${2:-test}"
BASE_RUN_NAME="${BASE_RUN_NAME:-base-eval}"
METRICS_DIR="${OUTPUT_ROOT:-outputs}/metrics"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[compare] Base run     : $BASE_RUN_NAME"
echo "[compare] Finetuned run: $FINETUNED_RUN_NAME"
echo "[compare] Split        : $SPLIT"

uv run python -m src.cli.compare \
  --base-run-name "$BASE_RUN_NAME" \
  --finetuned-run-name "$FINETUNED_RUN_NAME" \
  --split "$SPLIT" \
  --metrics-dir "$METRICS_DIR"

echo "[compare] Done. See $METRICS_DIR/comparisons/ and $METRICS_DIR/latest/"
