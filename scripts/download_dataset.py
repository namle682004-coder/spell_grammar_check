"""
Download a Hugging Face dataset repo (snapshot) to a local directory.

Usage:
    uv run python scripts/download_dataset.py \
        --repo-id ShynBui/Vietnamese_spelling_error \
        --local-dir data/raw/shynbui
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF dataset via snapshot_download.")
    parser.add_argument("--repo-id", default="ShynBui/Vietnamese_spelling_error")
    parser.add_argument("--local-dir", default="data/raw/shynbui")
    parser.add_argument("--repo-type", default="dataset")
    args = parser.parse_args()

    # Load .env if present
    env_path = Path(".env")
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
            print("[download] Loaded .env")
        except ImportError:
            # parse manually
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[download] Warning: HF_TOKEN not set. Private repos will fail.")

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] Repo    : {args.repo_id}")
    print(f"[download] Type    : {args.repo_type}")
    print(f"[download] LocalDir: {local_dir.resolve()}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[download] ERROR: huggingface_hub not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=str(local_dir),
        token=token,
    )

    print(f"[download] Done. Files saved to: {path}")

    # List downloaded files
    files = list(Path(path).iterdir())
    print(f"[download] Files ({len(files)}):")
    for f in sorted(files):
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name}  ({size // 1024} KB)")


if __name__ == "__main__":
    main()
