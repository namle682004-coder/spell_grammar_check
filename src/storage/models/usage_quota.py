from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from src.storage.base import Base, TimestampMixin, generate_uuid


class UsageQuota(Base, TimestampMixin):
    __tablename__ = "quotas"

    # ===== LOẠI 1: FIX CỨNG (server tự sinh) =====
    id = Column(String(16), primary_key=True, default=generate_uuid)

    # ===== LOẠI 2: BẮT BUỘC NHẬP (NOT NULL, không default) =====
    user_id = Column(String(16), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    period = Column(String(20), nullable=False)  # daily, weekly, monthly
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    request_limit = Column(Integer, nullable=False)
    token_limit = Column(Integer, nullable=False)
    cost_limit_usd = Column(Float, nullable=False)

    # ===== LOẠI 3: DEFAULT CỤ THỂ (NOT NULL, có default) =====
    requests_used = Column(Integer, nullable=False, default=0)
    tokens_used = Column(Integer, nullable=False, default=0)
    cost_used_usd = Column(Float, nullable=False, default=0.0)
    requests_last_minute = Column(Integer, nullable=False, default=0)
    requests_last_hour = Column(Integer, nullable=False, default=0)
    is_exceeded = Column(Boolean, nullable=False, default=False)
    exceeded_reason = Column(String(100), nullable=False, default="")

    # ===== LOẠI 4: CÓ THỂ NULL (nullable=True) =====
    minute_window_start = Column(DateTime, nullable=True)
    hour_window_start = Column(DateTime, nullable=True)
    exceeded_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="quotas")

    __table_args__ = (
        UniqueConstraint("user_id", "period", "period_start", name="uq_user_quota_period"),
        Index("idx_quotas_user_period", "user_id", "period", "period_start"),
    )

    def get_remaining_requests(self) -> int:
        return max(0, self.request_limit - self.requests_used)

    def get_remaining_tokens(self) -> int:
        return max(0, self.token_limit - self.tokens_used)
