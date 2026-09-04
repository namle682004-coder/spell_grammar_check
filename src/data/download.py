"""Download Multi-News dataset from various sources."""
from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.config import Config
from src.utils.hf_auth import login_hf


def download_dataset(cfg: Config) -> object:
    """
    Download dataset according to cfg.dataset.source_type.

    Returns a DatasetDict or dict of lists.
    """
    login_hf()
    source_type = cfg.dataset.source_type

    if source_type == "hf_dataset":
        return _load_hf_dataset(cfg)
    elif source_type == "hf_snapshot":
        return _load_hf_snapshot(cfg)
    elif source_type == "local":
        return _load_local(cfg)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def _load_hf_dataset(cfg: Config) -> object:
    import os
    from datasets import load_dataset

    repo_id = cfg.dataset.repo_id
    subset = cfg.dataset.get("subset", None)
    revision = cfg.dataset.get("revision", None)
    private = cfg.dataset.get("private", False)
    token = os.environ.get("HF_TOKEN") if private else None

    logger.info(
        f"Loading dataset '{repo_id}' (subset={subset}) from Hugging Face Hub...")
    ds = load_dataset(
        repo_id,
        subset,
        revision=revision,
        token=token,
        trust_remote_code=True,
    )
    logger.info(f"Dataset loaded: {ds}")
    return ds


def _load_hf_snapshot(cfg: Config) -> object:
    """Download a full repo snapshot (for custom private dataset repos)."""
    import os
    from huggingface_hub import snapshot_download
    from datasets import load_from_disk

    repo_id = cfg.dataset.repo_id
    raw_dir = Path(cfg.dataset.raw_dir) / repo_id.replace("/", "--")
    raw_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    logger.info(f"Downloading snapshot of '{repo_id}' to {raw_dir}...")
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(raw_dir),
        token=token,
    )
    logger.info(f"Snapshot saved to {local_path}. Loading from disk...")
    return load_from_disk(local_path)


def _load_local(cfg: Config) -> object:
    """Load dataset from local raw_dir JSON files or processed JSONL."""
    import json
    from pathlib import Path
    from datasets import Dataset, DatasetDict

    # Priority 1: raw_dir JSON files (train.json / validation.json / test.json)
    raw_dir = Path(cfg.dataset.raw_dir)
    # Keep canonical split names aligned with Hugging Face DatasetDict.
    raw_split_map = {"train": "train",
                     "validation": "validation", "test": "test"}
    raw_splits = {}
    for file_split, out_split in raw_split_map.items():
        p = raw_dir / f"{file_split}.json"
        if p.exists():
            records = json.loads(p.read_text())
            raw_splits[out_split] = Dataset.from_list(records)
            logger.info(f"Loaded {len(records)} records from {p}")
    if raw_splits:
        return DatasetDict(raw_splits)

    # Priority 2: processed JSONL
    processed_dir = Path(cfg.dataset.processed_dir)
    if (processed_dir / "dataset_dict.json").exists() or (processed_dir / "train").exists():
        from datasets import load_from_disk
        logger.info(f"Loading DatasetDict from {processed_dir}...")
        return load_from_disk(str(processed_dir))

    splits = {}

    # train/test names are stable.
    for split in ("train", "test"):
        p = processed_dir / f"{split}.jsonl"
        if p.exists():
            records = [json.loads(l)
                       for l in p.read_text().splitlines() if l.strip()]
            splits[split] = Dataset.from_list(records)

    # Backward-compatible validation detection: prefer validation.jsonl, fallback val.jsonl.
    validation_path = processed_dir / "validation.jsonl"
    if not validation_path.exists():
        validation_path = processed_dir / "val.jsonl"
    if validation_path.exists():
        records = [json.loads(
            l) for l in validation_path.read_text().splitlines() if l.strip()]
        splits["validation"] = Dataset.from_list(records)

    if splits:
        return DatasetDict(splits)

    raise FileNotFoundError(
        f"No dataset found in {raw_dir} or {processed_dir}")
