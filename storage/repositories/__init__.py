from src.storage.database import get_db_manager
from src.storage.repositories.user_repo import UserRepository
from src.storage.repositories.api_key_repo import ApiKeyRepository
from src.storage.repositories.request_repo import RequestRepository
from src.storage.repositories.correction_repo import CorrectionRepository
from src.storage.repositories.quota_repo import QuotaRepository
from src.storage.repositories.billing_repo import BillingRepository

class RepositoryFactory:
    """Factory để lấy các repository instances"""
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def __enter__(self):
        self.session = self.db_manager.SessionLocal()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()
    
    @property
    def users(self) -> UserRepository:
        return UserRepository(self.session)
    
    @property
    def api_keys(self) -> ApiKeyRepository:
        return ApiKeyRepository(self.session)
    
    @property
    def requests(self) -> RequestRepository:
        return RequestRepository(self.session)
    
    @property
    def corrections(self) -> CorrectionRepository:
        return CorrectionRepository(self.session)
    
    @property
    def quotas(self) -> QuotaRepository:
        return QuotaRepository(self.session)
    
    @property
    def billings(self) -> BillingRepository:
        return BillingRepository(self.session)

# Convenience function
def get_repository_factory():
    return RepositoryFactory()