"""Versioned artifact paths for train/eval runs (no silent overwrites)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.io import ensure_dir, write_json


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def is_versioned(cfg: Any) -> bool:
    save = cfg.save if hasattr(cfg, "save") else {}
    if hasattr(save, "get"):
        return bool(save.get("versioned", True))
    return bool(save.get("versioned", True) if isinstance(save, dict) else True)


def get_run_name(cfg: Any, default: str = "default") -> str:
    name = getattr(cfg, "run_name", None)
    if name:
        return str(name)
    return default


def get_run_id(cfg: Any) -> str:
    save = cfg.save
    explicit = save.get("run_id") if hasattr(save, "get") else None
    if explicit:
        return str(explicit)
    env_id = os.environ.get("RUN_ID") or os.environ.get("EVAL_RUN_ID")
    if env_id:
        return str(env_id)
    return new_run_id()


def ensure_run_id(cfg: Any, *, env_key: str = "RUN_ID") -> str:
    """Assign one run id per CLI invocation (generate + metrics share it)."""
    save = cfg.save
    explicit = save.get("run_id") if hasattr(save, "get") else None
    if explicit:
        os.environ[env_key] = str(explicit)
        return str(explicit)
    if os.environ.get(env_key):
        return str(os.environ[env_key])
    run_id = new_run_id()
    os.environ[env_key] = run_id
    return run_id


@dataclass(frozen=True)
class ArtifactRun:
    run_name: str
    run_id: str
    metrics_root: Path
    predictions_root: Path

    @classmethod
    def from_config(cls, cfg: Any) -> ArtifactRun:
        metrics_root = Path(cfg.save.metrics_dir)
        predictions_root = Path(cfg.save.predictions_dir)
        return cls(
            run_name=get_run_name(cfg),
            run_id=get_run_id(cfg),
            metrics_root=metrics_root,
            predictions_root=predictions_root,
        )

    @property
    def metrics_run_dir(self) -> Path:
        return self.metrics_root / "runs" / self.run_name / self.run_id

    @property
    def predictions_run_dir(self) -> Path:
        return self.predictions_root / "runs" / self.run_name / self.run_id

    def metrics_path(self, prefix: str, split: str) -> Path:
        return self.metrics_run_dir / f"{prefix}_{split}.json"

    def predictions_path(self, prefix: str, split: str) -> Path:
        return self.predictions_run_dir / f"{prefix}_{split}.jsonl"

    def manifest_path(self) -> Path:
        return self.metrics_run_dir / "manifest.json"


def flat_metrics_path(cfg: Any, prefix: str, split: str) -> Path:
    return Path(cfg.save.metrics_dir) / f"{prefix}_{split}.json"


def flat_predictions_path(cfg: Any, prefix: str, split: str) -> Path:
    return Path(cfg.save.predictions_dir) / f"{prefix}_{split}.jsonl"


def resolve_metrics_path(cfg: Any, prefix: str, split: str) -> Path:
    if is_versioned(cfg):
        return ArtifactRun.from_config(cfg).metrics_path(prefix, split)
    return flat_metrics_path(cfg, prefix, split)


def resolve_predictions_path(cfg: Any, prefix: str, split: str) -> Path:
    if is_versioned(cfg):
        return ArtifactRun.from_config(cfg).predictions_path(prefix, split)
    return flat_predictions_path(cfg, prefix, split)


def get_reference_text(record: dict) -> str:
    for key in ("completion", "clean_text", "target_summary"):
        val = record.get(key)
        if val:
            return str(val)
    return ""


def _latest_pointer_path(
    metrics_root: Path,
    run_name: str,
    prefix: str,
    split: str,
) -> Path:
    return metrics_root / "latest" / run_name / f"{prefix}_{split}.json"


def update_latest_pointer(
    *,
    metrics_root: Path,
    run_name: str,
    run_id: str,
    prefix: str,
    split: str,
    artifact_path: Path,
    extra: dict | None = None,
) -> Path:
    pointer = _latest_pointer_path(metrics_root, run_name, prefix, split)
    payload = {
        "run_name": run_name,
        "run_id": run_id,
        "prefix": prefix,
        "split": split,
        "path": str(artifact_path),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        **(extra or {}),
    }
    write_json(payload, pointer)

    run_latest = metrics_root / "runs" / run_name / "latest.json"
    write_json(
        {
            "run_id": run_id,
            "updated_at": payload["updated_at"],
            "metrics_dir": str(artifact_path.parent),
        },
        run_latest,
    )
    return pointer


def read_latest_metrics_path(
    metrics_root: str | Path,
    run_name: str,
    prefix: str,
    split: str,
) -> Path | None:
    pointer = _latest_pointer_path(Path(metrics_root), run_name, prefix, split)
    if not pointer.exists():
        return None
    data = json.loads(pointer.read_text())
    path = Path(data["path"])
    return path if path.exists() else None


def write_run_manifest(
    artifact: ArtifactRun,
    *,
    kind: str,
    prefix: str,
    split: str,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Path:
    manifest = {
        "kind": kind,
        "run_name": artifact.run_name,
        "run_id": artifact.run_id,
        "prefix": prefix,
        "split": split,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        **(extra or {}),
    }
    path = artifact.manifest_path()
    write_json(manifest, path)
    return path


def promote_finetune_model(
    source: str | Path,
    *,
    production_dir: str | Path = "outputs/production",
) -> Path:
    """Point outputs/production/current at a trained adapter/checkpoint."""
    source = Path(source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Model path not found: {source}")

    prod = ensure_dir(production_dir)
    current = prod / "current"
    if current.is_symlink() or current.is_file():
        current.unlink()
    elif current.is_dir() and not current.is_symlink():
        raise RuntimeError(
            f"{current} is a real directory; remove it before promoting."
        )

    current.symlink_to(source, target_is_directory=source.is_dir())

    manifest = {
        "source": str(source),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "current": str(current),
    }
    write_json(manifest, prod / "manifest.json")
    logger.info(f"Promoted production model: {current} -> {source}")
    return current


def resolve_finetune_model_path() -> str:
    """Path used by API inference (env > production > legacy)."""
    if env_path := os.environ.get("FINETUNE_MODEL"):
        return env_path

    production = Path("outputs/production/current")
    if production.exists():
        return str(production.resolve())

    legacy = (
        "outputs/adapters/"
        "vietnamese-grammar-qwen1.5b-shynbui-lora/final_model"
    )
    return legacy
