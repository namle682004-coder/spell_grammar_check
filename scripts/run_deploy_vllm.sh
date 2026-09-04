#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/deploy.yaml}"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[deploy] Running vLLM deploy with config: $CONFIG"
uv run python -m src.cli.deploy --config "$CONFIG"
