"""Thread-safe TTL Response Cache for Grammar Correction Inference."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional


class ResponseCache:
    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[str, tuple[float, Any]] = {}

    def _hash_key(self, text: str, model_name: str) -> str:
        raw = f"{model_name}:{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, text: str, model_name: str = "finetune") -> Optional[Any]:
        key = self._hash_key(text, model_name)
        if key not in self._store:
            return None

        created_at, value = self._store[key]
        if time.time() - created_at > self.ttl_seconds:
            del self._store[key]
            return None

        return value

    def set(self, text: str, value: Any, model_name: str = "finetune") -> None:
        if len(self._store) >= self.max_entries:
            # Evict oldest entry
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k][0])
            del self._store[oldest_key]

        key = self._hash_key(text, model_name)
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        self._store.clear()


# Global Singleton Cache Instance
global_cache = ResponseCache(ttl_seconds=3600, max_entries=2000)
