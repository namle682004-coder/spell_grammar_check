from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.storage.base import Base, TimestampMixin, generate_uuid


class SpellGrammarRequest(Base, TimestampMixin):
    __tablename__ = "requests"

    # ===== LOẠI 1: FIX CỨNG (server tự sinh) =====
    id = Column(String(16), primary_key=True, default=generate_uuid)

    # ===== LOẠI 2: BẮT BUỘC NHẬP (NOT NULL, không default) =====
    request_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(16), ForeignKey(
        "users.id"), nullable=False, index=True)
    # spell_check, grammar_check, full_check, batch_check
    request_type = Column(String(50), nullable=False)
    input_text = Column(Text, nullable=False)
    input_chars = Column(Integer, nullable=False)
    model_used = Column(String(100), nullable=False)
    processing_time_ms = Column(Integer, nullable=False, default=0)

    # ===== LOẠI 3: DEFAULT CỤ THỂ (NOT NULL, có default) =====
    input_tokens = Column(Integer, nullable=False, default=0)
    output_text = Column(Text, nullable=False, default="")
    output_chars = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    total_corrections = Column(Integer, nullable=False, default=0)
    corrections_spelling = Column(Integer, nullable=False, default=0)
    corrections_grammar = Column(Integer, nullable=False, default=0)
    corrections_punctuation = Column(Integer, nullable=False, default=0)
    corrections_style = Column(Integer, nullable=False, default=0)
    queue_wait_ms = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False, default="pending", index=True)
    status_code = Column(Integer, nullable=False, default=0)
    error_message = Column(Text, nullable=False, default="")
    error_code = Column(String(100), nullable=False, default="")
    extra_metadata = Column(JSONB, nullable=False, default={})

    # ===== LOẠI 4: CÓ THỂ NULL (nullable=True) =====
    api_key_id = Column(String(16), ForeignKey(
        "api_keys.id"), nullable=True, index=True)
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    session_id = Column(String(100), nullable=True, index=True)

    # Relationships
    user = relationship("User", back_populates="requests")
    api_key = relationship("ApiKey", back_populates="requests")
    corrections = relationship(
        "CorrectionDetail", back_populates="request", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_requests_user_created", "user_id", "created_at"),
        Index("idx_requests_status_created", "status", "created_at"),
        Index("idx_requests_session", "session_id"),
    )
