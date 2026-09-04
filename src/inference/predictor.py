import gc
import time
from dataclasses import dataclass

import torch
from unsloth import FastLanguageModel

from src.utils.artifacts import resolve_finetune_model_path

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

FINETUNE_MODEL = resolve_finetune_model_path()

current_model_name = None
model = None
tokenizer = None


@dataclass
class GenerationResult:
    output: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    latency_ms: float


def load_model(model_name: str):
    global model
    global tokenizer
    global current_model_name

    if current_model_name == model_name:
        return

    if model is not None:
        del model
        del tokenizer

        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=128,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)

    current_model_name = model_name


def generate(
    text: str,
    model_type: str = "finetune",
) -> str:
    return generate_with_usage(text=text, model_type=model_type).output


def generate_with_usage(
    text: str,
    model_type: str = "finetune",
) -> GenerationResult:
    if model_type == "base":
        model_name = BASE_MODEL
    else:
        model_name = FINETUNE_MODEL

    load_model(model_name)

    prompt = f"""Sửa lỗi chính tả và ngữ pháp cho đoạn văn sau. Giữ nguyên ý nghĩa:

{text}

Đoạn đã sửa:
"""

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    ).to(model.device)

    prompt_tokens = int(inputs.input_ids.shape[-1])
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
        )

    latency_ms = (time.time() - start) * 1000
    total_generated = int(outputs.shape[-1])
    completion_tokens = max(total_generated - prompt_tokens, 0)

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return GenerationResult(
        output=decoded,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        model_name=model_name,
        latency_ms=latency_ms,
    )
