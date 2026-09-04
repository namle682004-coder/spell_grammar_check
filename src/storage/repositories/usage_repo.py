
"""PostgreSQL usage and quota persistence."""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.storage.database import session_scope
from src.storage.models.spell_grammar_request import SpellGrammarRequest
from src.storage.models.usage_quota import UsageQuota
from src.storage.models.user import User


class UsageRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_user_by_id(self, user_id: str) -> User | None:
        return self.session.scalar(select(User).where(User.id == user_id))

    def get_or_create_user(self, email: str) -> User:
        user = self.session.scalar(select(User).where(User.email == email))
        if user is not None:
            return user
        user = User(email=email)
        self.session.add(user)
        self.session.flush()
        return user

    def get_quota_by_user_id(self, user_id: str) -> UsageQuota | None:
        now = datetime.utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        return self.session.scalar(
            select(UsageQuota).where(
                UsageQuota.user_id == user_id,
                UsageQuota.period == "monthly",
                UsageQuota.period_start == month_start,
            )
        )

    def create_quota(self, **kwargs) -> UsageQuota:
        quota = UsageQuota(**kwargs)
        self.session.add(quota)
        self.session.flush()
        return quota

    def update_quota(self, quota_id: str, **kwargs) -> UsageQuota | None:
        quota = self.session.get(UsageQuota, quota_id)
        if quota:
            for key, value in kwargs.items():
                setattr(quota, key, value)
            self.session.flush()
        return quota

    def increment_quota_usage(
        self, quota: UsageQuota, requests: int = 1, tokens: int = 0, cost: float = 0.0
    ) -> UsageQuota:
        quota.requests_used += requests
        quota.tokens_used += tokens
        quota.cost_used_usd += cost
        self.session.flush()
        return quota

    def log_request(
        self,
        user: User,
        input_text: str,
        output_text: str | None,
        model_used: str,
        request_type: str = "full_check",
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        processing_time_ms: float | None = None,
        status: str = "success",
        error_code: str | None = None,
        api_key_id: str | None = None,
    ) -> SpellGrammarRequest:
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        row = SpellGrammarRequest(
            request_id=request_id,
            user_id=user.id,
            api_key_id=api_key_id,
            request_type=request_type,
            input_text=input_text,
            input_chars=len(input_text),
            input_tokens=input_tokens or 0,
            output_text=output_text or "",
            output_chars=len(output_text or ""),
            output_tokens=output_tokens or 0,
            model_used=model_used,
            processing_time_ms=int(processing_time_ms or 0),
            status=status,
            error_code=error_code or "",
        )
        self.session.add(row)
        self.session.flush()
        return row


def get_usage_db_repository() -> UsageRepository:
    """Return a repository bound to a new session scope (caller must use within session)."""
    raise NotImplementedError(
        "Use UsageRepository(session) with get_db_manager().get_session() instead."
    )


def with_session(fn):
    """Decorator helper for one-shot DB operations."""

    def wrapper(*args, **kwargs):
        with session_scope() as session:
            return fn(session, *args, **kwargs)

    return wrapper
