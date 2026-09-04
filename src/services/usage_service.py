
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from src.storage.database import get_db_manager
from src.storage.repositories import (
    UserRepository,
    ApiKeyRepository,
    RequestRepository,
    CorrectionRepository
)

class UsageService:
    """Service for managing usage tracking and quotas"""
    
    @staticmethod
    def get_user_usage_stats(user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        db = get_db_manager()
        
        with db.get_session() as session:
            request_repo = RequestRepository(session)
            user_repo = UserRepository(session)
            api_key_repo = ApiKeyRepository(session)
            
            # Get user info
            user = user_repo.get_by_id(user_id)
            if not user:
                return {"error": "User not found"}
            
            # Get all API keys for user
            api_keys = api_key_repo.get_all(user_id=user_id)
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get all requests (lấy hết, filter bằng Python cho đơn giản)
            all_requests = request_repo.get_all(user_id=user_id)
            
            # Filter by date range and status
            successful = [
                r for r in all_requests 
                if r.status == "success" and r.created_at and r.created_at >= start_date
            ]
            
            # Filter by date range for total
            total_requests = len([r for r in all_requests if r.created_at and r.created_at >= start_date])
            
            # Calculate totals
            successful_requests = len(successful)
            total_tokens = sum((r.input_tokens or 0) + (r.output_tokens or 0) for r in successful)
            total_cost = sum(r.cost_usd or 0 for r in successful)
            total_corrections = sum(r.total_corrections or 0 for r in successful)
            
            # Calculate average processing time
            avg_time = sum(r.processing_time_ms or 0 for r in successful) / max(len(successful), 1)
            
            # Breakdown by request type
            breakdown = {}
            type_counts = {}
            for r in successful:
                req_type = r.request_type or "unknown"
                if req_type not in type_counts:
                    type_counts[req_type] = {"count": 0, "total_time": 0, "total_corrections": 0}
                type_counts[req_type]["count"] += 1
                type_counts[req_type]["total_time"] += r.processing_time_ms or 0
                type_counts[req_type]["total_corrections"] += r.total_corrections or 0
            
            for req_type, data in type_counts.items():
                breakdown[req_type] = {
                    "count": data["count"],
                    "avg_processing_ms": data["total_time"] / data["count"] if data["count"] > 0 else 0,
                    "total_corrections": data["total_corrections"]
                }
            
            # Daily usage
            daily_usage = {}
            for r in successful:
                day = r.created_at.date().isoformat() if r.created_at else None
                if day:
                    if day not in daily_usage:
                        daily_usage[day] = {"requests": 0, "tokens": 0, "cost": 0}
                    daily_usage[day]["requests"] += 1
                    daily_usage[day]["tokens"] += (r.input_tokens or 0) + (r.output_tokens or 0)
                    daily_usage[day]["cost"] += r.cost_usd or 0
            
            return {
                "user_id": user_id,
                "username": user.username,
                "period_days": days,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                "total_tokens": total_tokens,
                "total_cost_usd": round(total_cost, 6),
                "total_corrections": total_corrections,
                "avg_processing_time_ms": round(avg_time, 2),
                "breakdown_by_type": breakdown,
                "daily_usage": daily_usage,
                "api_keys_count": len(api_keys)
            }
    
    @staticmethod
    def get_api_key_stats(api_key_id: str) -> Dict[str, Any]:
        """Get statistics for a specific API key"""
        db = get_db_manager()
        
        with db.get_session() as session:
            api_key_repo = ApiKeyRepository(session)
            request_repo = RequestRepository(session)
            
            api_key = api_key_repo.get_by_id(api_key_id)
            if not api_key:
                return {"error": "API key not found"}
            
            # Get all requests for this API key
            requests = request_repo.get_all(api_key_id=api_key_id)
            successful = [r for r in requests if r.status == "success"]
            
            return {
                "api_key_id": api_key_id,
                "key_name": api_key.key_name,
                "total_requests": api_key.total_requests,
                "total_tokens": api_key.total_tokens,
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat() if api_key.created_at else None,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "rate_limits": {
                    "per_minute": api_key.rate_limit_per_minute,
                    "per_hour": api_key.rate_limit_per_hour,
                    "per_day": api_key.rate_limit_per_day
                },
                "recent_requests_count": len([r for r in requests if r.created_at and r.created_at > datetime.utcnow() - timedelta(hours=24)]),
                "success_rate": len(successful) / len(requests) if requests else 0
            }
    
    @staticmethod
    def get_quota_info(user_id: str) -> Dict[str, Any]:
        """Get quota information for a user"""
        db = get_db_manager()
        
        with db.get_session() as session:
            user_repo = UserRepository(session)
            request_repo = RequestRepository(session)
            
            user = user_repo.get_by_id(user_id)
            if not user:
                return {"error": "User not found"}
            
            # Define quota limits based on user role
            quota_limits = {
                "free": {
                    "requests_per_day": 100,
                    "requests_per_month": 1000,
                    "tokens_per_month": 100000,
                    "cost_limit_usd": 1.0
                },
                "pro": {
                    "requests_per_day": 1000,
                    "requests_per_month": 10000,
                    "tokens_per_month": 1000000,
                    "cost_limit_usd": 10.0
                },
                "enterprise": {
                    "requests_per_day": 10000,
                    "requests_per_month": 100000,
                    "tokens_per_month": 10000000,
                    "cost_limit_usd": 100.0
                }
            }
            
            limits = quota_limits.get(user.role, quota_limits["free"])
            
            # Get current usage this month
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Get all requests
            all_requests = request_repo.get_all(user_id=user_id)
            
            monthly_requests = len([
                r for r in all_requests 
                if r.status == "success" and r.created_at and r.created_at >= month_start
            ])
            
            daily_requests = len([
                r for r in all_requests 
                if r.status == "success" and r.created_at and r.created_at >= day_start
            ])
            
            # Calculate tokens used this month
            monthly_tokens = sum(
                (r.input_tokens or 0) + (r.output_tokens or 0) 
                for r in all_requests 
                if r.status == "success" and r.created_at and r.created_at >= month_start
            )
            
            return {
                "user_id": user_id,
                "role": user.role,
                "limits": limits,
                "current_usage": {
                    "today": {
                        "used": daily_requests,
                        "remaining": max(0, limits["requests_per_day"] - daily_requests),
                        "limit": limits["requests_per_day"]
                    },
                    "this_month": {
                        "requests": {
                            "used": monthly_requests,
                            "remaining": max(0, limits["requests_per_month"] - monthly_requests),
                            "limit": limits["requests_per_month"]
                        },
                        "tokens": {
                            "used": monthly_tokens,
                            "remaining": max(0, limits["tokens_per_month"] - monthly_tokens),
                            "limit": limits["tokens_per_month"]
                        },
                        "cost": {
                            "used": 0,
                            "limit": limits["cost_limit_usd"]
                        }
                    }
                },
                "is_over_quota": daily_requests >= limits["requests_per_day"] or monthly_requests >= limits["requests_per_month"]
            }
