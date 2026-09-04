"""CLI: promote a trained adapter to production/current."""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from src.utils.artifacts import promote_finetune_model
from src.utils.env import load_env
from src.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a finetuned model to outputs/production/current.",
    )
    parser.add_argument(
        "--source",
        help="Adapter/checkpoint directory (default: outputs/adapters/<run_name>/final_model)",
    )
    parser.add_argument(
        "--run-name",
        default="vietnamese-grammar-qwen1.5b-shynbui-lora",
        help="Run name under outputs/adapters/",
    )
    parser.add_argument(
        "--production-dir",
        default="outputs/production",
    )
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    load_env(args.env)
    setup_logging()

    source = args.source
    if not source:
        source = str(
            Path("outputs/adapters") / args.run_name / "final_model"
        )

    promote_finetune_model(source, production_dir=args.production_dir)
    logger.info(
        "Set FINETUNE_MODEL=outputs/production/current in .env (optional; auto-detected)."
    )


if __name__ == "__main__":
    main()
