"""Generate corrections with base or finetuned model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import Config
from src.inference.decoding import get_decoding_kwargs, generate_batch
from src.utils.artifacts import (
    ArtifactRun,
    get_reference_text,
    is_versioned,
    resolve_predictions_path,
    update_latest_pointer,
    write_run_manifest,
)
from src.utils.io import read_jsonl, write_jsonl, ensure_dir


def generate_with_base(cfg: Config) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(cfg.model.model_id)
    token = os.environ.get("HF_TOKEN")

    logger.info(f"Loading base model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token,
    )
    model.eval()

    _run_generation(model, tokenizer, cfg, prefix=str(cfg.save.prefix))


def _run_generation(model: Any, tokenizer: Any, cfg: Config, prefix: str) -> None:
    processed_dir = Path(cfg.dataset.processed_dir)
    decoding_kwargs = get_decoding_kwargs(cfg)
    batch_size = int(cfg.eval.get("batch_size", 4))
    versioned = is_versioned(cfg)
    artifact = ArtifactRun.from_config(cfg) if versioned else None

    splits = [s for s in cfg.eval.get("splits", ["val", "test"]).to_dict() if isinstance(s, str)] \
        if hasattr(cfg.eval.get("splits", None), "to_dict") \
        else list(cfg.eval.get("splits", ["val", "test"]))

    for split in splits:
        records = read_jsonl(processed_dir / f"{split}.jsonl")
        prompts = [r["prompt"] for r in records]
        references = [get_reference_text(r) for r in records]

        predictions: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            preds = generate_batch(model, tokenizer, batch_prompts, decoding_kwargs)
            predictions.extend(preds)
            logger.info(
                f"[{split}] Generated {min(i + batch_size, len(prompts))}/{len(prompts)}"
            )

        out_records = [
            {
                "id": r.get("id", str(i)),
                "prompt": p,
                "reference": ref,
                "prediction": pred,
                **(
                    {
                        "run_name": artifact.run_name,
                        "run_id": artifact.run_id,
                    }
                    if versioned and artifact is not None
                    else {}
                ),
            }
            for i, (r, p, ref, pred) in enumerate(
                zip(records, prompts, references, predictions)
            )
        ]

        out_path = resolve_predictions_path(cfg, prefix, split)
        ensure_dir(out_path.parent)
        write_jsonl(out_records, out_path)
        logger.info(f"Predictions saved to {out_path}")

        if versioned and artifact is not None:
            write_run_manifest(
                artifact,
                kind="predictions",
                prefix=prefix,
                split=split,
                extra={
                    "predictions_path": str(out_path),
                    "num_samples": len(out_records),
                    "model_id": str(cfg.model.get("model_id", "")),
                },
            )
            pointer = update_latest_pointer(
                metrics_root=Path(cfg.save.metrics_dir),
                run_name=artifact.run_name,
                run_id=artifact.run_id,
                prefix=prefix,
                split=split,
                artifact_path=out_path,
                extra={"artifact_type": "predictions"},
            )
            logger.info(f"Latest predictions pointer: {pointer}")
