"""Run full evaluation: load predictions, compute metrics, save."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config import Config
from src.evaluation.metrics import compute_all_metrics
from src.utils.artifacts import (
    ArtifactRun,
    is_versioned,
    update_latest_pointer,
    write_run_manifest,
)
from src.utils.io import read_jsonl, write_json, ensure_dir


def run_evaluation(cfg: Config) -> None:
    """Load predictions written by inference step and compute metrics."""
    pred_dir = Path(cfg.save.predictions_dir)
    metrics_dir = ensure_dir(cfg.save.metrics_dir)
    prefix = str(cfg.save.prefix)
    versioned = is_versioned(cfg)
    artifact = ArtifactRun.from_config(cfg) if versioned else None

    splits = list(cfg.eval.get("splits", ["val", "test"]))

    for split in splits:
        if versioned and artifact is not None:
            pred_path = artifact.predictions_path(prefix, split)
        else:
            pred_path = pred_dir / f"{prefix}_{split}.jsonl"

        if not pred_path.exists():
            logger.warning(f"Predictions not found: {pred_path}. Skipping.")
            continue

        records = read_jsonl(pred_path)
        predictions = [r["prediction"] for r in records]
        references = [r["reference"] for r in records]

        logger.info(
            f"Computing metrics for {prefix}/{split} "
            f"({len(records)} samples, versioned={versioned})..."
        )
        metrics = compute_all_metrics(predictions, references)
        metrics["num_samples"] = len(records)
        metrics["split"] = split
        metrics["prefix"] = prefix
        if versioned and artifact is not None:
            metrics["run_name"] = artifact.run_name
            metrics["run_id"] = artifact.run_id

        if versioned and artifact is not None:
            out_path = artifact.metrics_path(prefix, split)
            ensure_dir(artifact.metrics_run_dir)
        else:
            out_path = metrics_dir / f"{prefix}_{split}.json"

        write_json(metrics, out_path)
        logger.info(f"Metrics saved to {out_path}")

        if versioned and artifact is not None:
            write_run_manifest(
                artifact,
                kind="eval",
                prefix=prefix,
                split=split,
                metrics=metrics,
                extra={"predictions_path": str(pred_path)},
            )
            pointer = update_latest_pointer(
                metrics_root=Path(cfg.save.metrics_dir),
                run_name=artifact.run_name,
                run_id=artifact.run_id,
                prefix=prefix,
                split=split,
                artifact_path=out_path,
                extra={"predictions_path": str(pred_path)},
            )
            logger.info(f"Latest pointer updated: {pointer}")

        _log_summary(metrics)


def _log_summary(metrics: dict) -> None:
    keys = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "avg_pred_length_words"]
    logger.info("  ".join(f"{k}={metrics.get(k, 'N/A')}" for k in keys if k in metrics))
