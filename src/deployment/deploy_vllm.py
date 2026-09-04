"""Wrapper that launches the vLLM server for the merged model."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import httpx
from loguru import logger

from src.config import Config


def run_deploy(cfg: Config) -> None:
    model_path = str(cfg.model.merged_dir)
    host = str(cfg.deploy.host)
    port = int(cfg.deploy.port)
    gpu_mem = float(cfg.deploy.get("gpu_memory_utilization", 0.40))
    max_model_len = int(cfg.deploy.get("max_model_len", 256))
    extra_args = str(cfg.deploy.get("extra_args", ""))
    timeout = int(cfg.deploy.get("health_check_timeout", 120))

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Merged model not found at '{model_path}'. "
            "Run training with save.merge_16bit=true first."
        )

    cmd = [
        "vllm", "serve", model_path,
        "--host", host,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
        "--max-model-len", str(max_model_len),
        "--dtype", str(cfg.model.get("dtype", "float16")),
    ]
    if extra_args:
        cmd.extend(extra_args.split())

    logger.info(f"Launching vLLM: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)

    # Health check
    health_url = f"http://{host}:{port}/health"
    logger.info(f"Waiting for vLLM health at {health_url} (timeout={timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(health_url, timeout=5)
            if r.status_code == 200:
                logger.info("vLLM is healthy. Server ready.")
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        logger.warning("vLLM did not become healthy within timeout. Check logs.")

    logger.info(f"vLLM process running with PID {proc.pid}. Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        logger.info("vLLM process terminated.")
