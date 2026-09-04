#!/usr/bin/env bash
set -euo pipefail

echo "Running prompting test..."
uv run python src/cli/eval_prompting.py --config configs/eval_prompting.yaml