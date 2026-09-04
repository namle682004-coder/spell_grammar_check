"""Split dataset and export to JSONL files."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config import Config
from src.utils.io import ensure_dir, write_jsonl


def export_splits(
    splits: dict[str, list[dict]],
    cfg: Config,
) -> dict[str, Path]:
    """Write train/val/test JSONL files. Returns paths."""
    out_dir = Path(cfg.dataset.processed_dir)
    ensure_dir(out_dir)

    paths: dict[str, Path] = {}
    for split_name, records in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        write_jsonl(records, out_path)
        logger.info(f"Saved {len(records)} records to {out_path}")
        paths[split_name] = out_path

    return paths


def subsample(records: list, max_samples: int | None) -> list:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return records
    import random
    return random.sample(records, max_samples)
