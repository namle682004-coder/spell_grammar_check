"""CLI: compare base vs finetuned metrics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from src.utils.artifacts import read_latest_metrics_path, update_latest_pointer
from src.utils.env import load_env
from src.utils.logging import setup_logging
from src.utils.io import write_json
from src.evaluation.compare import compare_metrics, generate_report


def _resolve_metrics_path(
    explicit: str | None,
    metrics_dir: Path,
    run_name: str,
    prefix: str,
    split: str,
) -> Path:
    if explicit:
        return Path(explicit)
    latest = read_latest_metrics_path(metrics_dir, run_name, prefix, split)
    if latest:
        return latest
    flat = metrics_dir / f"{prefix}_{split}.json"
    if flat.exists():
        logger.warning(f"Using legacy flat metrics file: {flat}")
        return flat
    raise FileNotFoundError(
        f"No metrics for run_name={run_name!r} prefix={prefix!r} split={split!r}. "
        f"Run eval first or pass --base / --finetuned explicitly."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare base vs finetuned evaluation metrics.",
    )
    parser.add_argument("--base", help="Path to base metrics JSON (optional)")
    parser.add_argument("--finetuned", help="Path to finetuned metrics JSON (optional)")
    parser.add_argument(
        "--base-run-name",
        default="base-eval",
        help="Run name for latest base metrics pointer",
    )
    parser.add_argument(
        "--finetuned-run-name",
        default="vietnamese-grammar-qwen1.5b-shynbui-lora",
        help="Run name for latest finetuned metrics pointer",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--metrics-dir",
        default="outputs/metrics",
    )
    parser.add_argument("--output", help="Comparison JSON output path")
    parser.add_argument("--report", help="Markdown report output path")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    load_env(args.env)
    setup_logging()

    metrics_dir = Path(args.metrics_dir)
    base_path = _resolve_metrics_path(
        args.base,
        metrics_dir,
        args.base_run_name,
        "base",
        args.split,
    )
    finetuned_path = _resolve_metrics_path(
        args.finetuned,
        metrics_dir,
        args.finetuned_run_name,
        "finetuned",
        args.split,
    )

    logger.info(f"=== Compare | base={base_path} | finetuned={finetuned_path} ===")

    comparison = compare_metrics(str(base_path), str(finetuned_path))
    comparison["meta"] = {
        "base_path": str(base_path),
        "finetuned_path": str(finetuned_path),
        "split": args.split,
        "compared_at": datetime.now(timezone.utc).isoformat(),
    }

    if args.output:
        out_json = Path(args.output)
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_json = (
            metrics_dir
            / "comparisons"
            / f"{args.finetuned_run_name}_{args.split}_{run_id}.json"
        )

    if args.report:
        out_report = Path(args.report)
    else:
        out_report = out_json.with_suffix(".md")

    write_json(comparison, out_json)
    logger.info(f"Comparison JSON saved to {out_json}")

    generate_report(comparison, out_report)

    update_latest_pointer(
        metrics_root=metrics_dir,
        run_name="comparisons",
        run_id=args.finetuned_run_name,
        prefix="compare",
        split=args.split,
        artifact_path=out_json,
        extra={"report_path": str(out_report)},
    )
    logger.info(f"Latest comparison pointer: {metrics_dir}/latest/comparisons/compare_{args.split}.json")
    logger.info("=== Comparison complete ===")


if __name__ == "__main__":
    main()
