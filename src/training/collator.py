"""Dataset utilities for SFT training."""
from __future__ import annotations

from typing import Any

from datasets import Dataset
from loguru import logger


def make_hf_dataset(
    records: list[dict],
    text_column: str = "text",
) -> Dataset:
    """
    Convert processed records to HuggingFace Dataset.

    Expected input:
    [
        {
            "text": "...",
        }
    ]
    """

    cleaned_records = []

    for r in records:

        text = r.get(text_column, None)

        if not text:
            continue

        cleaned_records.append({
            text_column: text,
        })

    logger.info(
        f"Created HF dataset with "
        f"{len(cleaned_records)} samples"
    )

    return Dataset.from_list(cleaned_records)
