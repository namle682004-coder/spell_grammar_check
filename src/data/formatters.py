"""Format processed records into trainer-ready text/chat formats."""
from __future__ import annotations

from typing import Any


def format_as_text(record: dict, eos_token: str = "") -> str:
    """Single-text format: prompt + completion + EOS."""
    return record["prompt"] + record["completion"] + eos_token


def format_chat_template(record: dict, tokenizer: Any) -> str:
    """Apply tokenizer's chat template to messages list."""
    return tokenizer.apply_chat_template(
        record["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def apply_format(records: list[dict], style: str, tokenizer: Any | None = None) -> list[dict]:
    """Add 'text' field to each record according to style."""
    eos = tokenizer.eos_token if tokenizer else ""
    out = []
    for rec in records:
        r = dict(rec)
        if style == "instruction":
            r["text"] = format_as_text(rec, eos_token=eos)
        elif style == "chat" and tokenizer is not None:
            r["text"] = format_chat_template(rec, tokenizer)
        else:
            r["text"] = format_as_text(rec, eos_token=eos)
        out.append(r)
    return out
