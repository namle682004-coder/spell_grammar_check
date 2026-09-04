from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB
from src.storage.base import Base, TimestampMixin, generate_uuid


class PaymentTransaction(Base, TimestampMixin):
    __tablename__ = "payments"

    # ===== LOẠI 1: FIX CỨNG (server tự sinh) =====
    id = Column(String(16), primary_key=True, default=generate_uuid)

    # ===== LOẠI 2: BẮT BUỘC NHẬP (NOT NULL, không default) =====
    user_id = Column(String(16), ForeignKey(
        "users.id", ondelete="CASCADE"), nullable=False, index=True)
    transaction_id = Column(String(255), unique=True,
                            nullable=False, index=True)
    amount_usd = Column(Float, nullable=False)
    total_usd = Column(Float, nullable=False)

    # ===== LOẠI 3: DEFAULT CỤ THỂ (NOT NULL, có default) =====
    currency = Column(String(3), nullable=False, default="USD")
    tax_usd = Column(Float, nullable=False, default=0.0)
    discount_usd = Column(Float, nullable=False, default=0.0)
    status = Column(String(20), nullable=False, default="pending", index=True)
    extra_metadata = Column(JSONB, nullable=False, default={})
    payment_method = Column(String(20), nullable=False, default="")
    payment_provider = Column(String(50), nullable=False, default="")
    provider_payment_id = Column(String(255), nullable=False, default="")
    failure_reason = Column(Text, nullable=False, default="")
    description = Column(Text, nullable=False, default="")

    # ===== LOẠI 4: CÓ THỂ NULL (nullable=True) =====
    invoice_id = Column(String(255), unique=True, nullable=True)
    subscription_id = Column(String(255), nullable=True, index=True)
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    paid_at = Column(DateTime, nullable=True)
    refunded_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="payments")
