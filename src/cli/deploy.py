"""CLI: deploy finetuned model with vLLM."""
from __future__ import annotations

import argparse

from loguru import logger

from src.config import load_config
from src.utils.env import load_env
from src.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Deploy merged model with vLLM.")
    parser.add_argument("--config", required=True, help="Path to deploy.yaml")
    parser.add_argument("--env", default=".env")
    args = parser.parse_args()

    load_env(args.env)
    cfg = load_config(args.config, args.env)
    setup_logging()
    logger.info(f"=== Deploy | config={args.config} ===")

    from src.deployment.deploy_vllm import run_deploy
    run_deploy(cfg)


if __name__ == "__main__":
    main()
