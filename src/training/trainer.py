"""Build and configure TRL SFTTrainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger

from src.config import Config
from src.utils.io import ensure_dir


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    cfg: Config,
) -> Any:
    from trl import SFTConfig, SFTTrainer

    run_name = str(cfg.run_name)
    logs_dir = str(ensure_dir(cfg.save.logs_dir))
    adapter_dir = str(ensure_dir(Path(cfg.save.adapter_dir) / run_name))

    train_cfg = cfg.train
    per_device_train_batch_size = int(train_cfg.per_device_train_batch_size)
    per_device_eval_batch_size = int(
        train_cfg.get("per_device_eval_batch_size", 1))
    training_args = SFTConfig(
        output_dir=adapter_dir,
        run_name=run_name,

        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,

        gradient_accumulation_steps=int(
            train_cfg.gradient_accumulation_steps
        ),

        learning_rate=float(train_cfg.learning_rate),

        num_train_epochs=int(
            train_cfg.num_train_epochs
        ),

        eval_strategy="steps",
        eval_steps=int(train_cfg.eval_steps),

        save_strategy="steps",
        save_steps=int(train_cfg.save_steps),

        logging_steps=int(train_cfg.logging_steps),

        warmup_ratio=float(train_cfg.warmup_ratio),

        bf16=bool(train_cfg.bf16),

        gradient_checkpointing=bool(
            train_cfg.gradient_checkpointing
        ),

        dataloader_num_workers=int(
            train_cfg.get("dataloader_num_workers", 4)
        ),

        group_by_length=bool(
            train_cfg.get("group_by_length", True)
        ),

        report_to=str(
            train_cfg.get("report_to", "wandb")
        ),

        load_best_model_at_end=True,

        save_total_limit=3,

        logging_dir=str(Path(logs_dir) / run_name),

        packing=False,
    )
    logger.info(
        "Trainer batch sizes | train_per_device={} | eval_per_device={}",
        per_device_train_batch_size,
        per_device_eval_batch_size,
    )

    trainer = SFTTrainer(
        model=model,

        processing_class=tokenizer,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        args=training_args,

        dataset_text_field="text",

        max_seq_length=int(
            cfg.model.get("max_seq_length", 512)
        ),
    )
    return trainer
