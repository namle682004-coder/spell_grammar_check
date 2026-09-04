"""Compare base vs finetuned metrics and generate a report."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.utils.io import read_json


def compare_metrics(base_path: str, finetuned_path: str) -> dict:
    base = read_json(base_path)
    ft = read_json(finetuned_path)

    comparison: dict = {"base": {}, "finetuned": {}, "delta": {}}
    all_keys = set(base.keys()) | set(ft.keys())
    numeric_keys = [k for k in all_keys if isinstance(base.get(k), (int, float))]

    for k in numeric_keys:
        b_val = base.get(k, 0)
        f_val = ft.get(k, 0)
        comparison["base"][k] = b_val
        comparison["finetuned"][k] = f_val
        comparison["delta"][k] = round(f_val - b_val, 4)

    return comparison


def generate_report(comparison: dict, output_path: str | Path) -> None:
    lines = [
        "# Model Comparison Report",
        "",
        "| Metric | Base | Finetuned | Delta |",
        "|--------|------|-----------|-------|",
    ]
    keys = sorted(comparison["base"].keys())
    for k in keys:
        b = comparison["base"].get(k, "N/A")
        f = comparison["finetuned"].get(k, "N/A")
        d = comparison["delta"].get(k, "N/A")
        sign = "+" if isinstance(d, float) and d > 0 else ""
        lines.append(f"| {k} | {b} | {f} | {sign}{d} |")

    Path(output_path).write_text("\n".join(lines))
    logger.info(f"Report written to {output_path}")
