"""Persist usage records to storage backends (file or postgres)."""
from __future__ import annotations

import os
from datetime import date

from src.storage.context import RequestContext
from src.storage.paths import StorageLayout, get_storage_layout
from src.storage.schemas import (
    ApiRequestRecord,
    DailyAggregateRecord,
    ErrorRecord,
    LlmUsageRecord,
    UserProfileRecord,
    utc_now_iso,
)
from src.utils.io import append_jsonl, read_json, write_json


class FileUsageRepository:
    def __init__(self, layout: StorageLayout | None = None):
        self.layout = layout or get_storage_layout()

    def log_api_request(
        self,
        *,
        request_id: str,
        client_id: str,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        model_type: str | None = None,
        input_chars: int | None = None,
        output_chars: int | None = None,
        error_code: str | None = None,
    ) -> ApiRequestRecord:
        record = ApiRequestRecord(
            request_id=request_id,
            client_id=client_id,
            method=method,
            path=path,
            status_code=status_code,
            latency_ms=round(latency_ms, 2),
            model_type=model_type,
            input_chars=input_chars,
            output_chars=output_chars,
            error_code=error_code,
        )
        append_jsonl(record.model_dump(), self.layout.api_requests_file())
        self._touch_user(client_id, tokens=0)
        self._bump_daily_aggregate(
            path=path,
            prompt_tokens=0,
            completion_tokens=0,
            is_error=status_code >= 400,
            client_id=client_id,
        )
        return record

    def log_llm_usage(
        self,
        *,
        ctx: RequestContext,
        endpoint: str,
    ) -> LlmUsageRecord | None:
        if ctx.prompt_tokens is None or ctx.completion_tokens is None:
            return None

        record = LlmUsageRecord(
            request_id=ctx.request_id,
            client_id=ctx.client_id,
            model_name=ctx.model_name or "unknown",
            model_type=ctx.model_type or "unknown",
            endpoint=endpoint,
            prompt_tokens=ctx.prompt_tokens,
            completion_tokens=ctx.completion_tokens,
            total_tokens=ctx.total_tokens or (ctx.prompt_tokens + ctx.completion_tokens),
            latency_ms=ctx.llm_latency_ms,
        )
        append_jsonl(record.model_dump(), self.layout.llm_usage_file())
        self._touch_user(ctx.client_id, tokens=record.total_tokens)
        self._bump_daily_aggregate(
            path=endpoint,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            is_error=False,
            client_id=ctx.client_id,
        )
        return record

    def log_error(
        self,
        *,
        request_id: str,
        client_id: str,
        method: str,
        path: str,
        status_code: int,
        error_code: str,
        detail: str = "",
    ) -> ErrorRecord:
        record = ErrorRecord(
            request_id=request_id,
            client_id=client_id,
            method=method,
            path=path,
            status_code=status_code,
            error_code=error_code,
            detail=detail,
        )
        append_jsonl(record.model_dump(), self.layout.errors_file())
        self._bump_daily_aggregate(
            path=path,
            prompt_tokens=0,
            completion_tokens=0,
            is_error=True,
            client_id=client_id,
        )
        return record

    def _touch_user(self, client_id: str, *, tokens: int) -> None:
        path = self.layout.user_profile_file(client_id)
        if path.exists():
            data = read_json(path)
            profile = UserProfileRecord.model_validate(data)
            profile.total_requests += 1
            profile.total_tokens += tokens
            profile.last_seen_at = utc_now_iso()
        else:
            profile = UserProfileRecord(
                client_id=client_id,
                total_requests=1,
                total_tokens=tokens,
            )
        write_json(profile.model_dump(), path)

    def _bump_daily_aggregate(
        self,
        *,
        path: str,
        prompt_tokens: int,
        completion_tokens: int,
        is_error: bool,
        client_id: str,
    ) -> None:
        agg_path = self.layout.daily_aggregate_file()
        today = date.today().isoformat()
        if agg_path.exists():
            agg = DailyAggregateRecord.model_validate(read_json(agg_path))
        else:
            agg = DailyAggregateRecord(date=today)
        agg.total_requests += 1
        if is_error:
            agg.total_errors += 1
        agg.total_prompt_tokens += prompt_tokens
        agg.total_completion_tokens += completion_tokens
        write_json(agg.model_dump(), agg_path)


_repository: FileUsageRepository | None = None


def get_usage_repository() -> FileUsageRepository:
    global _repository
    backend = os.environ.get("USAGE_STORAGE_BACKEND", "file").lower()
    if backend == "postgres":
        raise NotImplementedError(
            "Postgres usage backend: wire UsageRepository in usage_middleware "
            "(src.storage.repositories.usage_repo)."
        )
    if _repository is None:
        _repository = FileUsageRepository()
    return _repository
