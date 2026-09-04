"""Generate summaries with finetuned model."""
from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from src.config import Config
from src.inference.generate_base import _run_generation


def generate_with_finetuned(cfg: Config) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = str(cfg.model.model_id)
    is_adapter = bool(cfg.model.get("is_adapter", False))
    base_model_id = cfg.model.get("base_model_id", None)
    token = os.environ.get("HF_TOKEN")

    logger.info(f"Loading finetuned model from: {model_id} (adapter={is_adapter})")
    # tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter and base_model_id:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            str(base_model_id),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=token,
        )
        model = PeftModel.from_pretrained(base, model_id)
        model = model.merge_and_unload()
    else:
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_id,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16,
        #     token=token,
        # )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            local_files_only=True
            )

    model.eval()
    _run_generation(model, tokenizer, cfg, prefix=str(cfg.save.prefix))
