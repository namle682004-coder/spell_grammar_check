"""Optional vLLM HTTP client for decoupled inference."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx


@dataclass
class VllmGenerationResult:
    output: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    latency_ms: float


def get_vllm_base_url() -> str | None:
    return os.getenv("VLLM_BASE_URL") or None


def generate_via_vllm(
    text: str,
    model_name: str = "finetune",
) -> VllmGenerationResult:
    base_url = get_vllm_base_url()
    if not base_url:
        raise RuntimeError("VLLM_BASE_URL is not configured")

    prompt = f"""Sửa lỗi chính tả và ngữ pháp cho đoạn văn sau. Giữ nguyên ý nghĩa:

{text}

Đoạn đã sửa:
"""
    start = time.time()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.1,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    latency_ms = (time.time() - start) * 1000
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    return VllmGenerationResult(
        output=choice,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model_name,
        latency_ms=latency_ms,
    )


async def generate_via_vllm_async(
    text: str,
    model_name: str = "finetune",
) -> VllmGenerationResult:
    base_url = get_vllm_base_url()
    if not base_url:
        raise RuntimeError("VLLM_BASE_URL is not configured")

    prompt = f"""Sửa lỗi chính tả và ngữ pháp cho đoạn văn sau. Giữ nguyên ý nghĩa:

{text}

Đoạn đã sửa:
"""
    start = time.time()
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0.1,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(f"{base_url.rstrip('/')}/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

    latency_ms = (time.time() - start) * 1000
    choice = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0))
    completion_tokens = int(usage.get("completion_tokens", 0))

    return VllmGenerationResult(
        output=choice,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model_name,
        latency_ms=latency_ms,
    )

