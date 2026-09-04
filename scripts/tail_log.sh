#!/usr/bin/env bash
LOG_DIR="${OUTPUT_ROOT:-outputs}/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/train_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
  echo "No training log found in $LOG_DIR"
  exit 1
fi

echo "[tail] Following: $LATEST_LOG"
tail -f "$LATEST_LOG"
