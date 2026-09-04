from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import or_
from src.storage.models.user import User, UserRole
from src.storage.repositories.base_repo import BaseRepository
import hashlib
import func

class UserRepository(BaseRepository[User]):
    def __init__(self, session: Session):
        super().__init__(User, session)
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Lấy user theo email"""
        return self.get_by(email=email)
    
    def get_by_username(self, username: str) -> Optional[User]:
        """Lấy user theo username"""
        return self.get_by(username=username)
    
    def search(self, query: str, skip: int = 0, limit: int = 100) -> List[User]:
        """Tìm kiếm user theo email hoặc username"""
        return self.session.query(User).filter(
            or_(
                User.email.ilike(f"%{query}%"),
                User.username.ilike(f"%{query}%")
            )
        ).offset(skip).limit(limit).all()
    
    def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """Lấy danh sách user đang active"""
        return self.get_all(skip=skip, limit=limit, is_active=True)
    
    def get_users_by_role(self, role: UserRole) -> List[User]:
        """Lấy user theo role"""
        return self.get_all(role=role)
    
    def update_last_active(self, user_id: str) -> Optional[User]:
        """Cập nhật thời gian hoạt động cuối"""
        from datetime import datetime
        return self.update(user_id, last_active_at=datetime.utcnow())
    
    def verify_user(self, user_id: str) -> Optional[User]:
        """Xác thực user"""
        from datetime import datetime
        return self.update(user_id, is_verified=True, verified_at=datetime.utcnow())
    
    def change_role(self, user_id: str, role: UserRole) -> Optional[User]:
        """Đổi role user"""
        return self.update(user_id, role=role)
    
    def get_stats(self, user_id: str) -> dict:
        """Lấy thống kê của user"""
        from src.storage.models.spell_grammar_request import SpellGrammarRequest, RequestStatus
        
        total_requests = self.session.query(SpellGrammarRequest).filter(
            SpellGrammarRequest.user_id == user_id
        ).count()
        
        successful_requests = self.session.query(SpellGrammarRequest).filter(
            SpellGrammarRequest.user_id == user_id,
            SpellGrammarRequest.status == RequestStatus.SUCCESS
        ).count()
        
        total_tokens = self.session.query(
            func.sum(SpellGrammarRequest.input_tokens + SpellGrammarRequest.output_tokens)
        ).filter(SpellGrammarRequest.user_id == user_id).scalar() or 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "total_tokens": total_tokens,
            "estimated_cost_usd": total_tokens * 0.001  # $0.001 per 1K tokens
        }