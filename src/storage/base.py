from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, DateTime, func
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid() -> str:
    """Generate UUID for primary keys"""
    return uuid.uuid4().hex[:16]

class TimestampMixin:
    """Mixin for created_at and updated_at"""
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

class SoftDeleteMixin:
    """Soft delete mixin"""
    is_deleted = Column(DateTime, default=False, nullable=False)
    deleted_at = Column(DateTime, nullable=True)