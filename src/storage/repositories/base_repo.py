from typing import Generic, TypeVar

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.storage.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    def __init__(self, model: type[ModelT], session: Session) -> None:
        self.model = model
        self.session = session

    def create(self, **kwargs) -> ModelT:
        instance = self.model(**kwargs)
        self.session.add(instance)
        self.session.flush()
        return instance

    def get_by_id(self, id: str) -> ModelT | None:
        return self.session.query(self.model).filter(self.model.id == id).first()

    def get_by(self, **filters) -> ModelT | None:
        query = self.session.query(self.model)
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.first()

    def get_all(self, skip: int = 0, limit: int = 100, **filters) -> list[ModelT]:
        query = self.session.query(self.model)
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.offset(skip).limit(limit).all()

    def update(self, id: str, **kwargs) -> ModelT | None:
        instance = self.get_by_id(id)
        if instance:
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            self.session.flush()
        return instance

    def delete(self, id: str) -> bool:
        instance = self.get_by_id(id)
        if instance:
            self.session.delete(instance)
            self.session.flush()
            return True
        return False

    def count(self, **filters) -> int:
        query = self.session.query(func.count(self.model.id))
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.scalar() or 0
