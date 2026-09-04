"""Central config loader: merges YAML file with environment overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class Config:
    """Dot-access wrapper around a nested dict loaded from YAML + env."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            val = self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no key '{name}'")
        return Config(val) if isinstance(val, dict) else val

    def __getitem__(self, key: str) -> Any:
        val = self._data[key]
        return Config(val) if isinstance(val, dict) else val

    def get(self, key: str, default: Any = None) -> Any:
        val = self._data.get(key, default)
        return Config(val) if isinstance(val, dict) else val

    def to_dict(self) -> dict:
        return self._data

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(config_path: str | Path, env_file: str | Path = ".env") -> Config:
    """Load YAML config and apply environment overrides."""
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path, override=False)

    with open(config_path) as f:
        data: dict = yaml.safe_load(f) or {}

    # Apply OUTPUT_ROOT override from env
    output_root = os.environ.get("OUTPUT_ROOT")
    if output_root and "save" in data:
        for key in ("adapter_dir", "merged_dir", "metrics_dir", "predictions_dir", "logs_dir"):
            if key in data["save"]:
                # replace leading 'outputs/' with output_root
                old = data["save"][key]
                rel = old.replace(
                    "outputs/", "", 1) if old.startswith("outputs/") else old
                data["save"][key] = str(Path(output_root) / rel)

    return Config(data)
