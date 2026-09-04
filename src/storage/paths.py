"""Paths for file-based usage storage under USAGE_STORAGE_ROOT."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class StorageLayout:
    root: Path

    def api_requests_file(self, day: date | None = None) -> Path:
        return self.root / "api_requests" / f"{_day(day)}.jsonl"

    def llm_usage_file(self, day: date | None = None) -> Path:
        return self.root / "llm_usage" / f"{_day(day)}.jsonl"

    def errors_file(self, day: date | None = None) -> Path:
        return self.root / "errors" / f"{_day(day)}.jsonl"

    def sessions_file(self, day: date | None = None) -> Path:
        return self.root / "sessions" / f"{_day(day)}.jsonl"

    def user_profile_file(self, client_id: str) -> Path:
        return self.root / "users" / f"{client_id}.json"

    def daily_aggregate_file(self, day: date | None = None) -> Path:
        return self.root / "aggregates" / "daily" / f"{_day(day)}.json"


def _day(day: date | None) -> str:
    return (day or date.today()).isoformat()


def get_storage_layout() -> StorageLayout:
    root = Path(os.environ.get("USAGE_STORAGE_ROOT", "./storage"))
    return StorageLayout(root=root)
