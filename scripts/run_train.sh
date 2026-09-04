#!/usr/bin/env bash

# export WANDB_API_KEY=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
# export WANDB_PROJECT=gramar_spell 
# export WANDB_ENTITY=namlthe187066-ew
# export WANDB_MODE=online
# export WANDB_DISABLED=false

set -euo pipefail
CONFIG="${1:-configs/train_finetune.yaml}"

if [ -f .env ]; then
  echo "[train] Loading .env..."
  set -a; source .env; set +a
fi

# extract run_name from config for log file naming
RUN_NAME=$(python3 -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('run_name','run'))" 2>/dev/null || echo "run")
LOG_DIR="${OUTPUT_ROOT:-outputs}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}.log"

echo "[train] Starting training in background..."
echo "[train] Config : $CONFIG"
echo "[train] Log    : $LOG_FILE"

nohup uv run python -m src.cli.train --config "$CONFIG" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "[train] PID    : $TRAIN_PID"
echo "$TRAIN_PID" > "$LOG_DIR/train_${RUN_NAME}.pid"

echo ""
echo "Monitor progress with:"
echo "  tail -f $LOG_FILE"
echo "  # or run: bash scripts/tail_log.sh"
