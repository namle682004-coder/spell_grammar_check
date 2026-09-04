"""Decoding profiles and generation helpers."""
from __future__ import annotations

from typing import Any

from src.config import Config

DECODING_PROFILES = {
    "benchmark_profile": {
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
    },
    "summary_profile": {
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
    },
}


def get_decoding_kwargs(cfg: Config) -> dict:
    """Return generation kwargs from config, merging profile defaults."""
    profile_name = str(cfg.eval.get("decoding_profile", "summary_profile"))

    # Check if config defines profiles block
    profiles_block = cfg.get("decoding_profiles", None)
    if profiles_block and profile_name in profiles_block.to_dict():
        base = dict(profiles_block[profile_name].to_dict())
    else:
        base = dict(DECODING_PROFILES.get(profile_name, DECODING_PROFILES["summary_profile"]))

    # Override with explicit eval keys
    for key in ("temperature", "top_p", "max_new_tokens"):
        val = cfg.eval.get(key, None)
        if val is not None:
            base[key] = val

    return base


def generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    decoding_kwargs: dict,
    device: str = "cuda",
) -> list[str]:
    """Run batched generation. Returns list of decoded strings."""
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(device)

    gen_kwargs = dict(decoding_kwargs)
    max_new = gen_kwargs.pop("max_new_tokens", 256)
    do_sample = gen_kwargs.pop("do_sample", True)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **gen_kwargs,
        )

    # Decode only the newly generated tokens
    input_lengths = inputs["input_ids"].shape[1]
    generated = outputs[:, input_lengths:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
