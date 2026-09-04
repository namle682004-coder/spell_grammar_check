"""Per-request context for API usage tracking."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    request_id: str
    client_id: str
    session_id: str | None = None
    model_type: str | None = None
    model_name: str | None = None
    input_chars: int | None = None
    output_chars: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    llm_latency_ms: float | None = None
    error_code: str | None = None
    extra: dict = field(default_factory=dict)


_ctx: ContextVar[RequestContext | None] = ContextVar("request_context", default=None)


def set_request_context(ctx: RequestContext) -> None:
    _ctx.set(ctx)


def get_request_context() -> RequestContext:
    ctx = _ctx.get()
    if ctx is None:
        raise RuntimeError("Request context is not set")
    return ctx


def reset_request_context() -> None:
    _ctx.set(None)
