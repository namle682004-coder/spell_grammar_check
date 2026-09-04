"""CLI: download + preprocess + split + export dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.config import load_config
from src.data.download import download_dataset
from src.data.manifest import compute_manifest, save_manifest
from src.data.preprocess import preprocess_sample
from src.data.split_export import export_splits, subsample
from src.utils.env import load_env
from src.utils.logging import setup_logging
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Download, preprocess and split Multi-News dataset.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--env", default=".env", help="Path to .env file")
    args = parser.parse_args()

    load_env(args.env)
    cfg = load_config(args.config, args.env)
    setup_logging()
    set_seed(int(cfg.seed))

    logger.info(f"=== Prepare Data | config={args.config} ===")

    # 1. Download
    raw_ds = download_dataset(cfg)

    # 2. Load tokenizer for truncation
    try:
        import os

        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(cfg.model.base_model_id),
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:
        logger.warning(
            f"Could not load tokenizer for truncation: {e}. Using word-count approximation.")
        tokenizer = None

    # 3. Preprocess all splits
    split_map = {"train": "train", "validation": "val", "test": "test"}
    processed: dict[str, list[dict]] = {}

    for raw_split, out_split in split_map.items():
        if raw_split not in raw_ds:
            logger.warning(f"Split '{raw_split}' not in dataset, skipping.")
            continue
        split_data = raw_ds[raw_split]
        records = []
        skipped = 0
        for sample in split_data:
            result = preprocess_sample(sample, cfg, tokenizer)
            if result is not None:
                records.append(result)
            else:
                skipped += 1
        logger.info(
            f"[{out_split}] Processed {len(records)} | Skipped {skipped}")
        processed[out_split] = records

    # 4. Subsample
    max_map = {
        "train": cfg.dataset.get("max_train_samples", None),
        "val": cfg.dataset.get("max_val_samples", None),
        "test": cfg.dataset.get("max_test_samples", None),
    }
    for split_name in list(processed.keys()):
        max_n = max_map.get(split_name, None)
        if max_n is not None:
            max_n = int(max_n)
        processed[split_name] = subsample(processed[split_name], max_n)
        logger.info(
            f"[{split_name}] Final size after subsample: {len(processed[split_name])}")

    # 5. Export JSONL
    export_splits(processed, cfg)

    # 6. Manifest
    manifest = compute_manifest(processed, tokenizer)
    save_manifest(manifest, Path(cfg.dataset.processed_dir) / "manifests")

    logger.info("=== Data preparation complete ===")


if __name__ == "__main__":
    main()
