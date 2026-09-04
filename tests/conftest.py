"""Shared pytest fixtures and mocks for CI (avoid loading GPU/ML deps)."""
import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies before any test imports src.main
for _mod in ("unsloth", "torch", "vllm"):
    sys.modules.setdefault(_mod, MagicMock())
