from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from src.storage.base import Base, TimestampMixin, generate_uuid


class CorrectionDetail(Base, TimestampMixin):
    __tablename__ = "corrections"

    # ===== LOẠI 1: FIX CỨNG (server tự sinh) =====
    id = Column(String(16), primary_key=True, default=generate_uuid)

    # ===== LOẠI 2: BẮT BUỘC NHẬP (NOT NULL, không default) =====
    request_id = Column(String(16), ForeignKey(
        "requests.id", ondelete="CASCADE"), nullable=False, index=True)
    # spelling, grammar, punctuation, style, typo
    correction_type = Column(String(20), nullable=False, index=True)
    original_text = Column(Text, nullable=False)
    corrected_text = Column(Text, nullable=False)

    # ===== LOẠI 3: DEFAULT CỤ THỂ (NOT NULL, có default) =====
    start_char = Column(Integer, nullable=False, default=0)
    end_char = Column(Integer, nullable=False, default=0)
    start_word = Column(Integer, nullable=False, default=0)
    end_word = Column(Integer, nullable=False, default=0)
    before_context = Column(Text, nullable=False, default="")
    after_context = Column(Text, nullable=False, default="")
    confidence = Column(Float, nullable=False, default=0.0)
    rule_id = Column(String(100), nullable=False, default="")
    suggestion = Column(Text, nullable=False, default="")
    user_ignored = Column(Boolean, nullable=False, default=False)
    is_learned = Column(Boolean, nullable=False, default=False)

    # ===== LOẠI 4: CÓ THỂ NULL (nullable=True) =====
    user_accepted = Column(Boolean, nullable=True)
    user_feedback_at = Column(DateTime, nullable=True)
    learned_at = Column(DateTime, nullable=True)

    # Relationships
    request = relationship("SpellGrammarRequest", back_populates="corrections")

    __table_args__ = (
        Index("idx_corrections_request_type", "request_id", "correction_type"),
        Index("idx_corrections_confidence", "confidence"),
    )
