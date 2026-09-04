from sqlalchemy.orm import Session

from src.storage.models.user import User
from src.storage.repositories.base_repo import BaseRepository


class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session):
        super().__init__(User, session)

    def get_by_email(self, email: str) -> User | None:
        return self.session.query(self.model).filter(self.model.email == email).first()

    def get_by_username(self, username: str) -> User | None:
        return self.session.query(self.model).filter(self.model.username == username).first()

    def get_all(self, skip: int = 0, limit: int = 100, **filters):
        query = self.session.query(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)
        return query.offset(skip).limit(limit).all()

    def count(self, **filters) -> int:
        from sqlalchemy import func

        query = self.session.query(func.count(self.model.id))
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)
        return query.scalar() or 0
