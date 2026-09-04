from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator

from src.storage.models import Base

class DatabaseManager:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError("DATABASE_URL is required")
        
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=os.getenv("SQL_ECHO", "false").lower() == "true"
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Context manager cho session - tự động commit/rollback"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Tạo tất cả bảng (dùng cho development)"""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Xóa tất cả bảng (dùng cho testing)"""
        Base.metadata.drop_all(bind=self.engine)
    
    def get_stats(self) -> dict:
        """Thống kê nhanh"""
        with self.get_session() as session:
            from sqlalchemy import func, text
            from src.storage.models.user import User
            from src.storage.models.spell_grammar_request import SpellGrammarRequest
            
            user_count = session.query(func.count(User.id)).scalar()
            request_count = session.query(func.count(SpellGrammarRequest.id)).scalar()
            return {
                "users": user_count,
                "requests": request_count,
                "database_url": self.database_url.split('@')[-1] if '@' in self.database_url else self.database_url
            }


# Singleton instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


# Helper functions for backward compatibility
@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for database session (compatibility layer)"""
    db = get_db_manager()
    with db.get_session() as session:
        yield session


def get_session() -> Session:
    """Get a new session (caller must close)"""
    db = get_db_manager()
    return db.SessionLocal()