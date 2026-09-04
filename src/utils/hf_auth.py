"""Hugging Face authentication helpers."""
from __future__ import annotations

import os

from loguru import logger


def login_hf(token: str | None = None) -> None:
    """Login to Hugging Face Hub if token is available."""
    tok = token or os.environ.get("HF_TOKEN")
    if not tok:
        logger.warning(
            "HF_TOKEN not set; private dataset/model access will fail.")
        return
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        logger.info("Logged in to Hugging Face Hub.")
    except Exception as e:
        logger.warning(f"HF login failed: {e}")
