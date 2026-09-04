"""Grammar correction wrapper for CLI tools."""
from __future__ import annotations

import re

from src.config import load_config
from src.inference.predictor import generate


class GrammarCorrector:
    """Zero-shot / prompting-based grammar correction using the loaded model."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.model_type = self.config.get("model_type", "base")

    def correct_single(self, text: str) -> str:
        output = generate(text=text, model_type=self.model_type)
        if "assistant" in output:
            output = output.split("assistant")[-1].strip()
        elif "Đoạn đã sửa:" in output:
            output = output.split("Đoạn đã sửa:")[-1].strip()
        output = re.sub(r"^[\n\s]+", "", output)
        output = re.sub(r"[\n\s]+$", "", output)
        return output.split("\n")[0].strip() if output else text
