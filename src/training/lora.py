"""Attach LoRA adapter using Unsloth or PEFT."""
from __future__ import annotations

from typing import Any

from loguru import logger

from src.config import Config


def attach_lora(model: Any, cfg: Config) -> Any:
    """
    Attach LoRA adapter.

    Priority:
    1. Unsloth FastLanguageModel
    2. PEFT fallback
    """

    r = int(cfg.lora.r)

    alpha = int(cfg.lora.alpha)

    dropout = float(cfg.lora.dropout)

    target_modules = list(cfg.lora.target_modules)

    bias = str(cfg.lora.get("bias", "none"))

    use_gc = cfg.lora.get(
        "use_gradient_checkpointing",
        "unsloth",
    )

    # =========================================================
    # Unsloth path
    # =========================================================
    try:
        from unsloth import FastLanguageModel

        logger.info("Attaching LoRA via Unsloth...")

        model = FastLanguageModel.get_peft_model(
            model,

            r=r,

            target_modules=target_modules,

            lora_alpha=alpha,

            lora_dropout=dropout,

            bias=bias,

            use_gradient_checkpointing=use_gc,

            random_state=42,

            use_rslora=bool(
                cfg.lora.get("use_rslora", False)
            ),

            loftq_config=cfg.lora.get(
                "loftq_config",
                None,
            ),
        )

        logger.info("LoRA attached via Unsloth.")

        return model

    except ImportError:
        logger.warning(
            "Unsloth not installed. "
            "Falling back to PEFT."
        )

    except Exception as e:
        logger.warning(
            f"Unsloth LoRA failed: {e}"
        )
        logger.warning(
            "Falling back to PEFT."
        )

    # =========================================================
    # PEFT fallback
    # =========================================================
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
    )

    logger.info("Attaching LoRA via PEFT...")

    lora_config = LoraConfig(
        r=r,

        lora_alpha=alpha,

        lora_dropout=dropout,

        target_modules=target_modules,

        bias=bias,

        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    logger.info("LoRA attached via PEFT.")

    return model
