import secrets
from typing import List

from sqlalchemy.orm import Session

from src.storage.models.api_key import ApiKey
from src.storage.repositories.base_repo import BaseRepository


class ApiKeyRepository(BaseRepository[ApiKey]):
    def __init__(self, session: Session):
        super().__init__(ApiKey, session)

    @staticmethod
    def generate_api_key() -> tuple:
        plain_key = f"sg_{secrets.token_urlsafe(32)}"
        hashed_key = secrets.token_urlsafe(32)
        prefix = plain_key[:12]
        return plain_key, hashed_key, prefix

    def get_by_user(self, user_id: str, active_only: bool = True) -> List[ApiKey]:
        query = self.session.query(self.model).filter(self.model.user_id == user_id)
        if active_only:
            query = query.filter(self.model.is_active == True)
        return query.all()

    def get_by_prefix(self, prefix: str) -> ApiKey | None:
        return self.get_by(key_prefix=prefix)

    def increment_usage(self, key_id: str, tokens: int = 0) -> None:
        api_key = self.get_by_id(key_id)
        if api_key:
            api_key.total_requests = (api_key.total_requests or 0) + 1
            api_key.total_tokens = (api_key.total_tokens or 0) + tokens
            self.session.flush()
