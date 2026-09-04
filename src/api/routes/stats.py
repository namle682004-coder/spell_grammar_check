from typing import Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_current_user
from src.services.usage_service import UsageService

router = APIRouter(prefix="/v1/stats", tags=["Statistics"])

class UsageStatsResponse(BaseModel):
    user_id: str
    username: str
    period_days: int
    total_requests: int
    successful_requests: int
    success_rate: float
    total_tokens: int
    total_cost_usd: float
    total_corrections: int
    avg_processing_time_ms: float
    breakdown_by_type: Dict
    daily_usage: Dict

class QuotaResponse(BaseModel):
    user_id: str
    role: str
    limits: Dict
    current_usage: Dict
    is_over_quota: bool

@router.get("/usage")
async def get_usage_stats(
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get usage statistics for current user"""
    result = UsageService.get_user_usage_stats(user["user_id"], days)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/quota")
async def get_quota_info(user: dict = Depends(get_current_user)):
    """Get quota information"""
    result = UsageService.get_quota_info(user["user_id"])
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/api-key/{api_key_id}")
async def get_api_key_stats(
    api_key_id: str,
    user: dict = Depends(get_current_user)
):
    """Get statistics for a specific API key"""
    result = UsageService.get_api_key_stats(api_key_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Check ownership
    if result.get("user_id") != user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return result

@router.get("/dashboard")
async def get_dashboard(user: dict = Depends(get_current_user)):
    """Get dashboard summary"""
    usage = UsageService.get_user_usage_stats(user["user_id"], 30)
    quota = UsageService.get_quota_info(user["user_id"])

    return {
        "user": {
            "user_id": user["user_id"],
            "username": usage.get("username"),
            "role": quota.get("role")
        },
        "usage_summary": {
            "total_requests": usage.get("total_requests", 0),
            "total_corrections": usage.get("total_corrections", 0),
            "total_cost": usage.get("total_cost_usd", 0),
            "avg_processing_time": usage.get("avg_processing_time_ms", 0)
        },
        "quota_status": {
            "today": quota.get("current_usage", {}).get("today", {}),
            "this_month": quota.get("current_usage", {}).get("this_month", {}),
            "is_over_quota": quota.get("is_over_quota", False)
        }
    }
