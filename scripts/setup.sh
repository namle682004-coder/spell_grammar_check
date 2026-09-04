#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Installing Python 3.10..."
uv python install 3.10

echo "[setup] Syncing dependencies..."
uv sync

echo "[setup] Downloading NLTK data..."
uv run python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

echo "[setup] Done. Copy .env.example to .env and fill in secrets."
