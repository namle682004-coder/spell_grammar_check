"""Pydantic schemas for usage log records."""
from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ApiRequestRecord(BaseModel):
    request_id: str
    client_id: str
    method: str
    path: str
    status_code: int
    latency_ms: float
    model_type: str | None = None
    input_chars: int | None = None
    output_chars: int | None = None
    error_code: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class LlmUsageRecord(BaseModel):
    request_id: str
    client_id: str
    model_name: str
    model_type: str
    endpoint: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float | None = None
    created_at: str = Field(default_factory=utc_now_iso)


class ErrorRecord(BaseModel):
    request_id: str
    client_id: str
    method: str
    path: str
    status_code: int
    error_code: str
    detail: str = ""
    created_at: str = Field(default_factory=utc_now_iso)


class UserProfileRecord(BaseModel):
    client_id: str
    total_requests: int = 0
    total_tokens: int = 0
    last_seen_at: str = Field(default_factory=utc_now_iso)


class DailyAggregateRecord(BaseModel):
    date: str
    total_requests: int = 0
    total_errors: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    unique_clients: int = 0
