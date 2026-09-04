"""CLI: generate predictions + compute metrics for a given model config."""
from __future__ import annotations

import argparse

from loguru import logger

from src.config import load_config
from src.utils.artifacts import ensure_run_id, is_versioned
from src.utils.env import load_env
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions and compute evaluation metrics.")
    parser.add_argument("--config", required=True,
                        help="eval_base.yaml or eval_finetuned.yaml")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    load_env(args.env)
    cfg = load_config(args.config, args.env)
    setup_logging()

    prefix = str(cfg.save.prefix)
    is_adapter = bool(cfg.model.get("is_adapter", False))
    if is_versioned(cfg):
        run_id = ensure_run_id(cfg, env_key="EVAL_RUN_ID")
        logger.info(f"Artifact run_id={run_id} (eval)")
    logger.info(f"=== Evaluate | prefix={prefix} | config={args.config} ===")

    # Generate predictions
    if prefix == "base":
        from src.inference.generate_base import generate_with_base
        generate_with_base(cfg)
    else:
        from src.inference.generate_finetuned import generate_with_finetuned
        generate_with_finetuned(cfg)

    # Compute metrics
    from src.evaluation.evaluate import run_evaluation
    run_evaluation(cfg)

    logger.info("=== Evaluation complete ===")


if __name__ == "__main__":
    main()
