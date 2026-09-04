"""Load base model and tokenizer with Unsloth or HF fallback."""
from __future__ import annotations

import os
from pyexpat import model
from typing import Any

from loguru import logger

from src.config import Config


def load_model_and_tokenizer(cfg: Config) -> tuple[Any, Any]:
    """
    Load model + tokenizer.

    Priority:
    1. Unsloth FastLanguageModel
    2. Standard HuggingFace fallback
    """

    model_id = str(cfg.model.base_model_id)

    load_in_4bit = bool(
        cfg.model.get("load_in_4bit", True)
    )

    max_seq_length = int(
        cfg.model.get("max_seq_length", 4096)
    )

    dtype_str = cfg.model.get("dtype", None)

    dtype = None

    if dtype_str and str(dtype_str) != "None":
        import torch
        dtype = getattr(torch, str(dtype_str), None)

    token = os.environ.get("HF_TOKEN")

    # =========================================================
    # Try Unsloth
    # =========================================================
    try:
        from unsloth import FastLanguageModel

        logger.info(
            f"Loading '{model_id}' via Unsloth "
            f"(4bit={load_in_4bit})..."
        )

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token,
        )

        # Training-safe config
        model.config.use_cache = False

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        logger.info("Model loaded via Unsloth.")

        return model, tokenizer

    except ImportError:
        logger.warning(
            "Unsloth not installed. "
            "Falling back to HuggingFace transformers."
        )

    except Exception as e:
        logger.warning(
            f"Unsloth load failed: {e}"
        )
        logger.warning(
            "Falling back to HuggingFace transformers."
        )

    # =========================================================
    # HF fallback
    # =========================================================
    return _load_hf(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        token=token,
    )


def _load_hf(
    model_id: str,
    load_in_4bit: bool,
    dtype: Any,
    token: str | None,
) -> tuple[Any, Any]:

    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    logger.info(
        f"Loading '{model_id}' via HuggingFace..."
    )

    # =========================================================
    # Tokenizer
    # =========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=token,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # =========================================================
    # Quantization
    # =========================================================
    quant_cfg = None
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.float16,
        token=token,
        trust_remote_code=True,
    )
    print(model)
    print(model.is_loaded_in_4bit)
    # Training-safe config
    model.config.use_cache = False

    # Resize embeddings if tokenizer changed
    model.resize_token_embeddings(len(tokenizer))

    logger.info("Model loaded via HuggingFace.")

    return model, tokenizer
