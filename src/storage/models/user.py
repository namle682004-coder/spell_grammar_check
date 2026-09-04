from sqlalchemy import Column, String, Boolean, Text, JSON, DateTime, Integer, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from src.storage.base import Base, TimestampMixin, generate_uuid

class User(Base, TimestampMixin):
    __tablename__ = "users"
    
    # ===== LOẠI 1: FIX CỨNG =====
    id = Column(String(16), primary_key=True, default=generate_uuid)
    
    # ===== LOẠI 2: BẮT BUỘC NHẬP =====
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    
    # ===== LOẠI 3: DEFAULT CỤ THỂ =====
    full_name = Column(String(255), nullable=False, default="")
    avatar_url = Column(Text, nullable=False, default="")
    language = Column(String(10), nullable=False, default="en")
    timezone = Column(String(50), nullable=False, default="UTC")
    role = Column(String(20), nullable=False, default="free")
    status = Column(String(20), nullable=False, default="active")
    email_verified = Column(Boolean, nullable=False, default=False)
    email_verified_at = Column(DateTime, nullable=True)  # nullable OK
    
    # 👇 SỬA: Thêm NOT NULL
    monthly_request_limit = Column(Integer, nullable=False, default=100)
    monthly_token_limit = Column(Integer, nullable=False, default=10000)
    monthly_budget_usd = Column(Float, nullable=False, default=1.0)
    
    default_model = Column(String(50), nullable=False, default="finetune")
    default_language = Column(String(10), nullable=False, default="en")
    auto_correct = Column(Boolean, nullable=False, default=True)
    show_explanations = Column(Boolean, nullable=False, default=True)
    
    # 👇 SỬA: default thành "" thay vì nullable
    stripe_customer_id = Column(String(255), nullable=False, default="")
    subscription_id = Column(String(255), nullable=False, default="")
    subscription_status = Column(String(50), nullable=False, default="inactive")
    subscription_end_at = Column(DateTime, nullable=True)  # nullable OK
    
    # Relationships
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    requests = relationship("SpellGrammarRequest", back_populates="user", cascade="all, delete-orphan")
    quotas = relationship("UsageQuota", back_populates="user", cascade="all, delete-orphan")
    payments = relationship("PaymentTransaction", back_populates="user", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "username": self.username,
            "full_name": self.full_name,
            "role": self.role,
            "status": self.status,
            "language": self.language,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }