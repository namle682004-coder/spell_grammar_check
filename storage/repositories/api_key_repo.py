from typing import Optional, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from src.storage.models.api_key import ApiKey
from src.storage.repositories.base_repo import BaseRepository
import secrets

class ApiKeyRepository(BaseRepository[ApiKey]):
    def __init__(self, session: Session):
        super().__init__(ApiKey, session)
    
    def create_api_key(self, user_id: str, key_name: str, **kwargs) -> tuple[ApiKey, str]:
        """Tạo API key mới, trả về (api_key_object, plain_key)"""
        plain_key, hashed_key, prefix = ApiKey.generate_api_key()
        
        api_key = self.create(
            user_id=user_id,
            key_name=key_name,
            api_key_hash=hashed_key,
            api_key_prefix=prefix,
            **kwargs
        )
        
        return api_key, plain_key
    
    def verify_api_key(self, plain_key: str) -> Optional[ApiKey]:
        """Xác thực API key (trong production nên hash rồi so sánh)"""
        # TODO: Implement proper hash comparison
        # For now, just check prefix and lookup
        prefix = plain_key[:8]
        
        api_key = self.session.query(ApiKey).filter(
            and_(
                ApiKey.api_key_prefix == prefix,
                ApiKey.is_active == True
            )
        ).first()
        
        if not api_key:
            return None
        
        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return None
        
        # Update last used
        self.update(api_key.id, last_used_at=datetime.utcnow())
        
        return api_key
    
    def get_user_keys(self, user_id: str, active_only: bool = True) -> List[ApiKey]:
        """Lấy tất cả API keys của user"""
        filters = {"user_id": user_id}
        if active_only:
            filters["is_active"] = True
        return self.get_all(**filters)
    
    def revoke_key(self, key_id: str) -> bool:
        """Thu hồi API key"""
        return self.update(key_id, is_active=False) is not None
    
    def update_rate_limits(self, key_id: str, per_minute: int = None, per_hour: int = None, per_day: int = None) -> Optional[ApiKey]:
        """Cập nhật rate limits cho API key"""
        updates = {}
        if per_minute is not None:
            updates['rate_limit_per_minute'] = per_minute
        if per_hour is not None:
            updates['rate_limit_per_hour'] = per_hour
        if per_day is not None:
            updates['rate_limit_per_day'] = per_day
        return self.update(key_id, **updates)
    
    def increment_usage(self, key_id: str, tokens: int = 0) -> None:
        """Tăng counter usage cho API key"""
        api_key = self.get_by_id(key_id)
        if api_key:
            api_key.total_requests += 1
            api_key.total_tokens += tokens
            api_key.last_used_at = datetime.utcnow()
            self.session.flush()
    
    def get_usage_stats(self, key_id: str, days: int = 30) -> dict:
        """Lấy thống kê usage của API key"""
        from src.storage.models.spell_grammar_request import SpellGrammarRequest
        
        since_date = datetime.utcnow() - timedelta(days=days)
        
        requests = self.session.query(SpellGrammarRequest).filter(
            SpellGrammarRequest.api_key_id == key_id,
            SpellGrammarRequest.created_at >= since_date
        ).all()
        
        total_requests = len(requests)
        total_tokens = sum(r.input_tokens or 0 + r.output_tokens or 0 for r in requests)
        total_cost = sum(r.cost_usd or 0 for r in requests)
        
        # Daily breakdown
        daily_stats = {}
        for req in requests:
            day = req.created_at.date().isoformat()
            if day not in daily_stats:
                daily_stats[day] = {"requests": 0, "tokens": 0, "cost": 0}
            daily_stats[day]["requests"] += 1
            daily_stats[day]["tokens"] += (req.input_tokens or 0) + (req.output_tokens or 0)
            daily_stats[day]["cost"] += req.cost_usd or 0
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "daily_breakdown": daily_stats,
            "period_days": days
        }