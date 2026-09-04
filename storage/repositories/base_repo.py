from typing import TypeVar, Generic, Type, List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from src.storage.base import Base
import uuid

T = TypeVar('T', bound=Base)

class BaseRepository(Generic[T]):
    """Base repository với các method CRUD cơ bản"""
    
    def __init__(self, model: Type[T], session: Session):
        self.model = model
        self.session = session
    
    def create(self, **kwargs) -> T:
        """Tạo mới một record"""
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.flush()
        return instance
    
    def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        """Tạo nhiều records"""
        instances = [self.model(**item) for item in items]
        self.session.add_all(instances)
        self.session.flush()
        return instances
    
    def get_by_id(self, id: str) -> Optional[T]:
        """Lấy record theo ID"""
        return self.session.query(self.model).filter(self.model.id == id).first()
    
    def get_by(self, **filters) -> Optional[T]:
        """Lấy record theo điều kiện"""
        query = self.session.query(self.model)
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.first()
    
    def get_all(self, skip: int = 0, limit: int = 100, **filters) -> List[T]:
        """Lấy nhiều records với phân trang"""
        query = self.session.query(self.model)
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.offset(skip).limit(limit).all()
    
    def update(self, id: str, **kwargs) -> Optional[T]:
        """Cập nhật record"""
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance
    
    def delete(self, id: str, soft_delete: bool = True) -> bool:
        """Xóa record (mềm hoặc cứng)"""
        instance = self.get_by_id(id)
        if not instance:
            return False
        
        if soft_delete and hasattr(instance, 'is_deleted'):
            instance.is_deleted = True
            if hasattr(instance, 'deleted_at'):
                from datetime import datetime
                instance.deleted_at = datetime.utcnow()
        else:
            self.session.delete(instance)
        
        self.session.flush()
        return True
    
    def count(self, **filters) -> int:
        """Đếm số records"""
        query = self.session.query(func.count(self.model.id))
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.scalar()
    
    def exists(self, **filters) -> bool:
        """Kiểm tra record tồn tại"""
        return self.count(**filters) > 0