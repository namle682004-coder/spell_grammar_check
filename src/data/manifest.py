"""Generate and save dataset manifest with statistics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.io import write_json


def compute_manifest(splits: dict[str, list[dict]], tokenizer: Any | None = None) -> dict:
    manifest: dict = {"splits": {}}
    for split_name, records in splits.items():
        src_lens = []
        tgt_lens = []
        for rec in records:
            src = rec.get("source_text", "")
            tgt = rec.get("target_summary", "")
            if tokenizer:
                src_lens.append(
                    len(tokenizer.encode(src, add_special_tokens=False)))
                tgt_lens.append(
                    len(tokenizer.encode(tgt, add_special_tokens=False)))
            else:
                src_lens.append(len(src.split()))
                tgt_lens.append(len(tgt.split()))

        def stats(vals: list) -> dict:
            if not vals:
                return {}
            return {
                "count": len(vals),
                "mean": round(sum(vals) / len(vals), 1),
                "min": min(vals),
                "max": max(vals),
            }

        manifest["splits"][split_name] = {
            "num_samples": len(records),
            "source_token_stats": stats(src_lens),
            "target_token_stats": stats(tgt_lens),
        }

    return manifest


def save_manifest(manifest: dict, output_dir: str | Path) -> Path:
    p = Path(output_dir) / "manifest.json"
    write_json(manifest, p)
    logger.info(f"Manifest saved to {p}")
    return p
