from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, func, BigInteger
from datetime import datetime
import uuid

Base = declarative_base()

class TimestampMixin:
    """Thời gian tạo và cập nhật"""
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

class SoftDeleteMixin:
    """Xóa mềm"""
    is_deleted = Column(DateTime, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)

def generate_id() -> str:
    """Tạo ID ngẫu nhiên"""
    return uuid.uuid4().hex[:16]