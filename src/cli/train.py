"""SFT training pipeline for Vietnamese grammar correction."""
from __future__ import annotations

import os
from pathlib import Path
import torch

from datasets import load_dataset
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel, UnslothTrainer

from src.config import Config
from src.training.model_loader import load_model_and_tokenizer
from src.utils.artifacts import (
    ArtifactRun,
    ensure_run_id,
    is_versioned,
    update_latest_pointer,
    write_run_manifest,
)
from src.utils.io import ensure_dir, write_json
from src.utils.seed import set_seed


def run_training(cfg: Config) -> None:
    set_seed(int(cfg.seed))
    if is_versioned(cfg):
        run_id = ensure_run_id(cfg, env_key="RUN_ID")
        logger.info(f"Artifact run_id={run_id} (train)")

    # -----------------------------
    # WandB
    # -----------------------------
    wandb_key = os.environ.get("WANDB_API_KEY")

    if wandb_key and cfg.train.get("report_to") == "wandb":
        import wandb

        wandb.login(key=wandb_key)

        wandb.init(
            project=os.environ.get(
                "WANDB_PROJECT",
                str(cfg.project_name),
            ),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=str(cfg.run_name),
            config=cfg.to_dict(),
        )

    # -----------------------------
    # Load dataset
    # -----------------------------
    processed_dir = Path(cfg.dataset.processed_dir)

    logger.info(f"Loading dataset from {processed_dir}")

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(processed_dir / "train.jsonl"),
            "validation": str(processed_dir / "val.jsonl"),
            "test": str(processed_dir / "test.jsonl"),
        },
    )

    train_ds = dataset["train"]
    val_ds = dataset["validation"]
    test_ds = dataset["test"]

    logger.info(
        f"FULL dataset | "
        f"Train={len(train_ds)} | "
        f"Val={len(val_ds)} | "
        f"Test={len(test_ds)}"
    )

    # -----------------------------
    # Optional sampling
    # -----------------------------
    train_limit = cfg.train.sample_train_size
    val_limit = cfg.train.sample_val_size
    test_limit = cfg.train.sample_test_size

    # logger.info(
    #     f"Sampling config | "
    #     f"train={train_limit} | "
    #     f"val={val_limit} | "
    #     f"test={test_limit}"
    # )

    if train_limit is not None:
        train_limit = min(int(train_limit), len(train_ds))
        train_ds = train_ds.shuffle(seed=cfg.seed).select(range(train_limit))

    if val_limit is not None:
        val_limit = min(int(val_limit), len(val_ds))
        val_ds = val_ds.shuffle(seed=cfg.seed).select(range(val_limit))

    if test_limit is not None:
        test_limit = min(int(test_limit), len(test_ds))
        test_ds = test_ds.shuffle(seed=cfg.seed).select(range(test_limit))

    logger.info(
        f"SAMPLED dataset | "
        f"Train={len(train_ds)} | "
        f"Val={len(val_ds)} | "
        f"Test={len(test_ds)}"
    )

    # -----------------------------
    # Model & Tokenizer
    # -----------------------------
    logger.info("Loading model and tokenizer...")

    model, tokenizer = load_model_and_tokenizer(cfg)

    # -----------------------------
    # Chat formatting
    # -----------------------------
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

        return {"text": text}

    train_ds = train_ds.map(format_chat)
    val_ds = val_ds.map(format_chat)
    test_ds = test_ds.map(format_chat)

    logger.info("Formatted chat template")

    # -----------------------------
    # LoRA
    # -----------------------------
    logger.info("Applying LoRA...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora.r,
        target_modules=cfg.lora.target_modules,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
    )

    # -----------------------------
    # Output dirs
    # -----------------------------
    output_dir = Path(cfg.save.adapter_dir) / cfg.run_name

    ensure_dir(output_dir)

    # -----------------------------
    # Training args
    # -----------------------------
    training_args = SFTConfig(
        output_dir=str(output_dir),

        run_name=cfg.run_name,

        per_device_train_batch_size=cfg.train.per_device_train_batch_size,

        per_device_eval_batch_size=cfg.train.get(
            "per_device_eval_batch_size",
            1,
        ),

        gradient_accumulation_steps=cfg.train.get(
            "gradient_accumulation_steps",
            4,
        ),

        learning_rate=float(cfg.train.learning_rate),

        num_train_epochs=cfg.train.num_train_epochs,

        logging_steps=cfg.train.get(
            "logging_steps",
            10,
        ),

        eval_steps=cfg.train.get(
            "eval_steps",
            200,
        ),

        save_steps=cfg.train.get(
            "save_steps",
            200,
        ),

        save_strategy="steps",

        eval_strategy="steps",

        bf16=cfg.train.get("bf16", True),

        fp16=cfg.train.get("fp16", False),

        warmup_ratio=cfg.train.get(
            "warmup_ratio",
            0.05,
        ),

        weight_decay=cfg.train.get(
            "weight_decay",
            0.01,
        ),

        lr_scheduler_type=cfg.train.get(
            "lr_scheduler_type",
            "cosine",
        ),

        report_to=cfg.train.get(
            "report_to",
            "none",
        ),

        logging_dir=str(
            ensure_dir(
                Path(cfg.save.logs_dir) / cfg.run_name
            )
        ),

        save_total_limit=2,

        load_best_model_at_end=False,

        dataset_text_field="text",

        max_length=cfg.model.get(
            "max_seq_length",
            512,
        ),

        packing=False,
    )

    # -----------------------------
    # Data collator
    # -----------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # -----------------------------
    # SFT config
    # -----------------------------
    sft_config = SFTConfig(
        output_dir=str(output_dir),

        run_name=cfg.run_name,

        per_device_train_batch_size=cfg.train.per_device_train_batch_size,

        per_device_eval_batch_size=cfg.train.get(
            "per_device_eval_batch_size",
            1,
        ),

        gradient_accumulation_steps=cfg.train.get(
            "gradient_accumulation_steps",
            4,
        ),

        learning_rate=float(cfg.train.learning_rate),

        num_train_epochs=cfg.train.num_train_epochs,

        logging_steps=cfg.train.get(
            "logging_steps",
            10,
        ),

        eval_steps=cfg.train.get(
            "eval_steps",
            200,
        ),

        save_steps=cfg.train.get(
            "save_steps",
            200,
        ),

        save_strategy="steps",

        eval_strategy="steps",

        bf16=cfg.train.get("bf16", True),

        fp16=cfg.train.get("fp16", False),

        warmup_ratio=cfg.train.get(
            "warmup_ratio",
            0.05,
        ),

        weight_decay=cfg.train.get(
            "weight_decay",
            0.01,
        ),

        lr_scheduler_type=cfg.train.get(
            "lr_scheduler_type",
            "cosine",
        ),

        report_to=cfg.train.get(
            "report_to",
            "none",
        ),

        logging_dir=str(
            ensure_dir(
                Path(cfg.save.logs_dir) / cfg.run_name
            )
        ),

        save_total_limit=2,

        load_best_model_at_end=False,
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = UnslothTrainer(
        model=model,

        processing_class=tokenizer,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        data_collator=data_collator,

        args=training_args,
    )

    # -----------------------------
    # Train
    # -----------------------------
    logger.info("Starting training...")

    trainer.train()

    # -----------------------------
    # Save
    # -----------------------------
    logger.info("Saving model...")

    final_dir = output_dir / "final_model"

    trainer.save_model(str(final_dir))

    tokenizer.save_pretrained(str(final_dir))

    # -----------------------------
    # Metrics (versioned — never overwrite prior runs)
    # -----------------------------
    train_payload = {"train_history": trainer.state.log_history}
    if is_versioned(cfg):
        artifact = ArtifactRun.from_config(cfg)
        metrics_path = artifact.metrics_run_dir / "train_metrics.json"
        ensure_dir(artifact.metrics_run_dir)
        write_json(train_payload, metrics_path)
        write_run_manifest(
            artifact,
            kind="train",
            prefix="train",
            split="summary",
            metrics={"train_loss": train_payload["train_history"][-1].get("train_loss")},
            extra={
                "adapter_dir": str(final_dir),
                "config_run_name": str(cfg.run_name),
            },
        )
        update_latest_pointer(
            metrics_root=Path(cfg.save.metrics_dir),
            run_name=artifact.run_name,
            run_id=artifact.run_id,
            prefix="train",
            split="metrics",
            artifact_path=metrics_path,
            extra={"adapter_dir": str(final_dir)},
        )
        legacy_dir = ensure_dir(Path(cfg.save.metrics_dir) / cfg.run_name)
        write_json(train_payload, legacy_dir / "metrics.json")
        logger.info(f"Train metrics (versioned): {metrics_path}")
    else:
        metrics_dir = ensure_dir(Path(cfg.save.metrics_dir) / cfg.run_name)
        write_json(train_payload, metrics_dir / "metrics.json")

    logger.success(f"Training complete! Saved to {final_dir}")
    logger.info(
        "Promote to API: uv run python -m src.cli.promote --run-name "
        f"{cfg.run_name}"
    )

    # -----------------------------
    # WandB finish
    # -----------------------------
    if wandb_key and cfg.train.get("report_to") == "wandb":
        import wandb

        wandb.finish()


if __name__ == "__main__":
    from src.config import load_config

    cfg = load_config("configs/train_finetune.yaml")

    run_training(cfg)
