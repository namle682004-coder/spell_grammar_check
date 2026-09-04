#!/usr/bin/env bash
# Download ShynBui/Vietnamese_spelling_error dataset to data/raw/
set -euo pipefail

REPO_ID="${1:-ShynBui/Vietnamese_spelling_error}"
LOCAL_DIR="data/raw/shynbui"

if [ -f .env ]; then
  set -a; source .env; set +a
fi

echo "[download] Dataset : $REPO_ID"
echo "[download] Local   : $LOCAL_DIR"

uv run python scripts/download_dataset.py --repo-id "$REPO_ID" --local-dir "$LOCAL_DIR"
