"""Core training pipeline: load → LoRA → train → save."""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from src.config import Config
from src.training.model_loader import load_model_and_tokenizer
from src.training.lora import attach_lora
from src.training.trainer import build_trainer
from src.training.collator import make_hf_dataset
from src.utils.artifacts import ArtifactRun, is_versioned, update_latest_pointer, write_run_manifest
from src.utils.io import ensure_dir, write_json, read_jsonl
from src.data.formatters import apply_format


def run_training(cfg: Config) -> None:
    import wandb
    from src.utils.seed import set_seed

    set_seed(int(cfg.seed))

    # W&B init
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", str(cfg.project_name)),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=str(cfg.run_name),
            config=cfg.to_dict(),
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Attach LoRA
    model = attach_lora(model, cfg)

    # Load data
    processed_dir = Path(cfg.dataset.processed_dir)
    logger.info(f"Loading processed splits from {processed_dir}")
    train_records = read_jsonl(processed_dir / "train.jsonl")
    val_records = read_jsonl(processed_dir / "val.jsonl")

    prompt_style = str(cfg.preprocess.get("prompt_style", "instruction"))
    train_records = apply_format(train_records, prompt_style, tokenizer)
    val_records = apply_format(val_records, prompt_style, tokenizer)

    train_ds = make_hf_dataset(train_records)
    val_ds = make_hf_dataset(val_records)
    logger.info(f"Train: {len(train_records)} | Val: {len(val_records)}")

    # Build trainer
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg)

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    logger.info(f"Training done. Metrics: {train_result.metrics}")

    # Save metrics (versioned when enabled in config)
    run_name = str(cfg.run_name)
    if is_versioned(cfg):
        artifact = ArtifactRun.from_config(cfg)
        metrics_path = artifact.metrics_run_dir / "train_summary.json"
        ensure_dir(artifact.metrics_run_dir)
        write_json(train_result.metrics, metrics_path)
        write_run_manifest(
            artifact,
            kind="train",
            prefix="train",
            split="summary",
            extra={"adapter_run_name": run_name},
        )
        update_latest_pointer(
            metrics_root=Path(cfg.save.metrics_dir),
            run_name=artifact.run_name,
            run_id=artifact.run_id,
            prefix="train",
            split="summary",
            artifact_path=metrics_path,
        )
    metrics_dir = ensure_dir(Path(cfg.save.metrics_dir) / run_name)
    write_json(train_result.metrics, metrics_dir / "train_summary.json")

    # Save adapter
    adapter_path = Path(cfg.save.adapter_dir) / run_name
    ensure_dir(adapter_path)
    trainer.save_model(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    logger.info(f"Adapter saved to {adapter_path}")

    # Export merged_16bit if configured
    if bool(cfg.save.get("merge_16bit", True)):
        _export_merged(model, tokenizer, cfg)

    if wandb_key:
        wandb.finish()


def _export_merged(model: Any, tokenizer: Any, cfg: Config) -> None:
    run_name = str(cfg.run_name)
    merged_path = Path(cfg.save.merged_dir) / run_name
    ensure_dir(merged_path)
    max_mem = float(cfg.save.get("maximum_memory_usage", 0.7))
    logger.info(f"Merging and saving 16-bit model to {merged_path}...")
    try:
        from unsloth import FastLanguageModel
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
            maximum_memory_usage=max_mem,
        )
    except Exception:
        # Fallback: standard HF save after merging PEFT
        from peft import PeftModel
        logger.warning("Unsloth merge failed. Trying PEFT merge_and_unload...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
    logger.info(f"Merged model saved to {merged_path}")


# Provide type annotation for Any used without import
from typing import Any  # noqa: E402
