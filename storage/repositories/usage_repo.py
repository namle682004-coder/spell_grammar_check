from typing import Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
from src.storage.models.usage_quota import UsageQuota, QuotaPeriod
from src.storage.repositories.base_repo import BaseRepository

class QuotaRepository(BaseRepository[UsageQuota]):
    def __init__(self, session: Session):
        super().__init__(UsageQuota, session)
    
    def get_or_create_quota(self, user_id: str, period: QuotaPeriod, api_key_id: str = None) -> UsageQuota:
        """Lấy hoặc tạo quota cho user"""
        now = datetime.utcnow()
        
        # Tính period_start và period_end
        if period == QuotaPeriod.DAILY:
            period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=1)
        elif period == QuotaPeriod.WEEKLY:
            period_start = now - timedelta(days=now.weekday())
            period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start + timedelta(days=7)
        elif period == QuotaPeriod.MONTHLY:
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = (period_start + timedelta(days=32)).replace(day=1)
        else:  # YEARLY
            period_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            period_end = period_start.replace(year=period_start.year + 1)
        
        # Tìm quota hiện có
        quota = self.session.query(UsageQuota).filter(
            and_(
                UsageQuota.user_id == user_id,
                UsageQuota.quota_period == period,
                UsageQuota.period_start == period_start
            )
        ).first()
        
        if not quota:
            # Lấy limits từ user plan
            from src.storage.models.user import UserRole
            user = self.session.query(User).filter(User.id == user_id).first()
            
            if user.role == "free":
                request_limit = 100
                token_limit = 10000
                cost_limit = 1.0
            elif user.role == "pro":
                request_limit = 10000
                token_limit = 1000000
                cost_limit = 50.0
            else:  # enterprise
                request_limit = 100000
                token_limit = 10000000
                cost_limit = 500.0
            
            quota = self.create(
                user_id=user_id,
                api_key_id=api_key_id,
                quota_period=period,
                period_start=period_start,
                period_end=period_end,
                request_limit=request_limit,
                token_limit=token_limit,
                cost_limit_usd=cost_limit
            )
        
        return quota
    
    def check_quota(self, user_id: str, tokens: int = 0, cost: float = 0.0) -> bool:
        """Kiểm tra xem user còn quota không"""
        daily_quota = self.get_or_create_quota(user_id, QuotaPeriod.DAILY)
        
        if daily_quota.is_over_request_limit():
            return False
        if daily_quota.is_over_token_limit():
            return False
        if daily_quota.is_over_cost_limit():
            return False
        
        return True
    
    def consume_quota(self, user_id: str, requests: int = 1, tokens: int = 0, cost: float = 0.0) -> bool:
        """Trừ quota"""
        daily_quota = self.get_or_create_quota(user_id, QuotaPeriod.DAILY)
        
        daily_quota.requests_used += requests
        daily_quota.tokens_used += tokens
        daily_quota.cost_used_usd += cost
        
        if daily_quota.is_over_request_limit() or daily_quota.is_over_token_limit() or daily_quota.is_over_cost_limit():
            daily_quota.is_exceeded = True
            daily_quota.exceeded_at = datetime.utcnow()
        
        self.session.flush()
        return not daily_quota.is_exceeded
    
    def reset_quotas(self) -> int:
        """Reset tất cả quotas hết hạn"""
        now = datetime.utcnow()
        expired = self.session.query(UsageQuota).filter(
            UsageQuota.period_end < now
        ).all()
        
        count = len(expired)
        for quota in expired:
            self.session.delete(quota)
        
        self.session.flush()
        return count