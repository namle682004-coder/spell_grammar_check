"""I/O helpers: JSON, JSONL, directory management."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def write_jsonl(records: Iterable[dict], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
