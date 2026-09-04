from sqlalchemy.orm import Session

from src.storage.models.correction_detail import CorrectionDetail
from src.storage.repositories.base_repo import BaseRepository


class CorrectionRepository(BaseRepository[CorrectionDetail]):
    def __init__(self, session: Session):
        super().__init__(CorrectionDetail, session)

    def get_by_request(self, request_id: str) -> list[CorrectionDetail]:
        return (
            self.session.query(self.model)
            .filter(self.model.request_id == request_id)
            .all()
        )

    def get_all(self, **filters) -> list[CorrectionDetail]:
        query = self.session.query(self.model)
        for key, value in filters.items():
            query = query.filter(getattr(self.model, key) == value)
        return query.all()
