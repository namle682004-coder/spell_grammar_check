from sqlalchemy.orm import Session

from src.storage.models.spell_grammar_request import SpellGrammarRequest
from src.storage.repositories.base_repo import BaseRepository


class RequestRepository(BaseRepository[SpellGrammarRequest]):
    def __init__(self, session: Session):
        super().__init__(SpellGrammarRequest, session)

    def get_by_request_id(self, request_id: str) -> SpellGrammarRequest | None:
        return (
            self.session.query(self.model)
            .filter(self.model.request_id == request_id)
            .first()
        )

    def get_all(self, skip: int = 0, limit: int = 100, **filters):
        query = self.session.query(self.model)
        for key, value in filters.items():
            if hasattr(self.model, key):
                query = query.filter(getattr(self.model, key) == value)
        return query.offset(skip).limit(limit).all()

    def get_user_requests(self, user_id: str, limit: int = 100, offset: int = 0):
        return (
            self.session.query(self.model)
            .filter(self.model.user_id == user_id)
            .order_by(self.model.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )
