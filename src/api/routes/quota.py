from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from src.api.dependencies import get_current_user
from src.services.usage_service import UsageService
from src.storage.database import get_db_manager
from src.storage.repositories import UsageRepository

router = APIRouter(prefix="/v1/quota", tags=["Quota Management"])

class QuotaCheckResponse(BaseModel):
    allowed: bool
    remaining_today: int
    remaining_this_month: int
    message: str

@router.get("/check")
async def check_quota(
    tokens: int = 0,
    user: dict = Depends(get_current_user)
):
    """Check if user has remaining quota"""
    result = UsageService.get_quota_info(user["user_id"])
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    today_remaining = result.get("current_usage", {}).get("today", {}).get("remaining", 0)
    month_remaining = result.get("current_usage", {}).get("this_month", {}).get("requests", {}).get("remaining", 0)
    
    return QuotaCheckResponse(
        allowed=today_remaining > 0 and month_remaining > 0,
        remaining_today=today_remaining,
        remaining_this_month=month_remaining,
        message=f"Remaining: {today_remaining} requests today, {month_remaining} this month"
    )

@router.get("/limits")
async def get_limits(user: dict = Depends(get_current_user)):
    """Get quota limits for current user"""
    result = UsageService.get_quota_info(user["user_id"])
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    return {
        "role": result.get("role"),
        "limits": result.get("limits"),
        "current_usage": result.get("current_usage")
    }