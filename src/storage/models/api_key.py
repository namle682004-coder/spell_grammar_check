import secrets

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from src.storage.base import Base, TimestampMixin, generate_uuid


class ApiKey(Base, TimestampMixin):
    __tablename__ = "api_keys"

    # ===== LOẠI 1: FIX CỨNG =====
    id = Column(String(16), primary_key=True, default=generate_uuid)

    # ===== LOẠI 2: BẮT BUỘC NHẬP =====
    user_id = Column(String(16), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_name = Column(String(100), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False)
    key_prefix = Column(String(10), nullable=False)

    # ===== LOẠI 3: DEFAULT CỤ THỂ =====
    permissions = Column(JSON, nullable=False, default={
        "spell_check": True,
        "grammar_check": True,
        "style_check": True,
        "batch_process": False
    })

    # 👇 SỬA: Giảm rate limits cho free user
    rate_limit_per_second = Column(Integer, nullable=False, default=2)      # 2 req/giây
    rate_limit_per_minute = Column(Integer, nullable=False, default=30)     # 30 req/phút
    rate_limit_per_hour = Column(Integer, nullable=False, default=500)      # 500 req/giờ
    rate_limit_per_day = Column(Integer, nullable=False, default=2000)      # 2000 req/ngày

    total_requests = Column(Integer, nullable=False, default=0)
    total_tokens = Column(Integer, nullable=False, default=0)
    total_cost_usd = Column(Float, nullable=False, default=0.0)
    is_active = Column(Boolean, nullable=False, default=True)

    # ===== LOẠI 4: CÓ THỂ NULL =====
    last_used_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")
    requests = relationship("SpellGrammarRequest", back_populates="api_key")

    @staticmethod
    def generate() -> tuple:
        plain_key = f"sg_{secrets.token_urlsafe(32)}"
        key_hash = secrets.token_urlsafe(32)
        prefix = plain_key[:12]
        return plain_key, key_hash, prefix
