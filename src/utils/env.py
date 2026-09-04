"""Environment helpers."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_env(env_file: str | Path = ".env") -> None:
    """Load .env file if present."""
    path = Path(env_file)
    if path.exists():
        load_dotenv(path, override=False)


def require_env(key: str) -> str:
    """Return env var or raise with a clear message."""
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(
            f"Environment variable '{key}' is required but not set. "
            f"Check your .env file."
        )
    return val


def get_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN")


def get_wandb_key() -> str | None:
    return os.environ.get("WANDB_API_KEY")


def get_api_key() -> str | None:
    return os.environ.get("API_KEY")
