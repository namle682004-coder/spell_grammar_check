"""Grammar / spelling correction preprocessing."""

from __future__ import annotations

import re
from typing import Any

from src.config import Config


def preprocess_sample(
    sample: dict,
    cfg: Config,
    tokenizer: Any | None = None,
) -> dict | None:
    """
    Convert raw dataset sample into SFT format.

    Expected dataset format (ShynBui):
    {
        "text": "Câu đúng",
        "error_text": "câu bị lỗi"
    }

    Train direction:
        noisy_text -> clean_text
    """

    # =========================================================
    # INPUT / TARGET
    # =========================================================

    clean_text: str = str(sample.get("text", "")).strip()
    noisy_text: str = str(sample.get("error_text", "")).strip()

    # Skip invalid samples
    if not clean_text or not noisy_text:
        return None

    # =========================================================
    # CLEAN TEXT
    # =========================================================

    clean_text = _clean_text(clean_text)
    noisy_text = _clean_text(noisy_text)

    # =========================================================
    # TRUNCATION
    # =========================================================

    max_source_tokens = int(
        cfg.preprocess.get("max_source_tokens", 256)
    )

    max_target_tokens = int(
        cfg.preprocess.get("max_target_tokens", 256)
    )

    truncation_strategy = str(
        cfg.preprocess.get("truncation_strategy", "tail")
    )

    if tokenizer is not None:
        noisy_text = _truncate_text(
            noisy_text,
            max_source_tokens,
            truncation_strategy,
            tokenizer,
        )

        clean_text = _truncate_text(
            clean_text,
            max_target_tokens,
            truncation_strategy,
            tokenizer,
        )

    # =========================================================
    # PROMPT TEMPLATE
    # =========================================================

    instruction_tmpl = str(
        cfg.preprocess.instruction_template
    )

    user_content = instruction_tmpl.format(
        input=noisy_text,
        source_text=noisy_text,
        error_text=noisy_text,
    )

    # =========================================================
    # RETURN SFT FORMAT
    # =========================================================

    return {
        "id": sample.get("id", ""),

        # raw fields
        "clean_text": clean_text,
        "error_text": noisy_text,

        # train fields
        "prompt": user_content,
        "completion": clean_text,

        # chat format
        "messages": [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": clean_text,
            },
        ],
    }


def _clean_text(text: str) -> str:
    """
    Basic text normalization.
    """

    if not text:
        return ""

    text = text.strip()

    # normalize spaces
    text = re.sub(r"[ \t]{2,}", " ", text)

    # normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def _truncate_text(
    text: str,
    max_tokens: int,
    strategy: str,
    tokenizer: Any,
) -> str:
    """
    Truncate text using tokenizer.
    """

    if not text:
        return ""

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if len(tokens) <= max_tokens:
        return text

    if strategy == "head":
        tokens = tokens[:max_tokens]

    elif strategy == "tail":
        tokens = tokens[-max_tokens:]

    elif strategy == "head_tail":
        half = max_tokens // 2
        tokens = tokens[:half] + tokens[-half:]

    else:
        tokens = tokens[:max_tokens]

    return tokenizer.decode(
        tokens,
        skip_special_tokens=True,
    )
