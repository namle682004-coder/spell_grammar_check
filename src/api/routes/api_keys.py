from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.services.auth_service import AuthService
from src.storage.database import get_db_manager
from src.storage.repositories import ApiKeyRepository

router = APIRouter(prefix="/v1/api-keys", tags=["API Keys"])

class CreateApiKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    rate_limit_per_minute: Optional[int] = 60
    rate_limit_per_hour: Optional[int] = 1000
    rate_limit_per_day: Optional[int] = 10000

class CreateApiKeyResponse(BaseModel):
    success: bool
    api_key_id: Optional[str] = None
    api_key: Optional[str] = None
    key_name: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None

@router.get("/")
async def list_api_keys(user: dict = Depends(get_current_user)):
    """List all API keys for current user"""
    result = AuthService.list_api_keys(user["user_id"])
    return result

@router.post("/create", response_model=CreateApiKeyResponse)
async def create_api_key(
    req: CreateApiKeyRequest,
    user: dict = Depends(get_current_user)
):
    """Create a new API key"""
    from src.services.auth_service import AuthService as Auth

    db = get_db_manager()

    with db.get_session() as session:
        api_key_repo = ApiKeyRepository(session)

        plain_key, hashed_key, prefix = Auth.generate_api_key()

        api_key = api_key_repo.create(
            user_id=user["user_id"],
            key_name=req.name,
            key_hash=hashed_key,
            key_prefix=prefix,
            rate_limit_per_minute=req.rate_limit_per_minute,
            rate_limit_per_hour=req.rate_limit_per_hour,
            rate_limit_per_day=req.rate_limit_per_day,
            is_active=True
        )

        return CreateApiKeyResponse(
            success=True,
            api_key_id=api_key.id,
            api_key=plain_key,
            key_name=req.name,
            message="API key created successfully. Save this key - it won't be shown again!"
        )

@router.delete("/{api_key_id}")
async def delete_api_key(
    api_key_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete/Revoke an API key"""
    result = AuthService.revoke_api_key(api_key_id, user["user_id"])
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.patch("/{api_key_id}/toggle")
async def toggle_api_key(
    api_key_id: str,
    active: bool = True,
    user: dict = Depends(get_current_user)
):
    """Enable or disable an API key"""
    db = get_db_manager()

    with db.get_session() as session:
        api_key_repo = ApiKeyRepository(session)

        api_key = api_key_repo.get_by_id(api_key_id)
        if not api_key or api_key.user_id != user["user_id"]:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key_repo.update(api_key_id, is_active=active)

        return {
            "success": True,
            "api_key_id": api_key_id,
            "is_active": active,
            "message": f"API key {'enabled' if active else 'disabled'}"
        }
