from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user_from_jwt
from src.services.auth_service import AuthService

router = APIRouter(prefix="/v1/auth", tags=["Authentication"])

class RegisterRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "user@example.com"})
    username: str = Field(..., min_length=3, max_length=50, json_schema_extra={"example": "john_doe"})
    password: str = Field(..., min_length=6, json_schema_extra={"example": "securepassword"})

class LoginRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "user@example.com"})
    password: str = Field(..., json_schema_extra={"example": "securepassword"})

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str = Field(..., min_length=6)

class RevokeKeyRequest(BaseModel):
    api_key_id: str

class AuthResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
    code: Optional[str] = None

class ApiKeysResponse(BaseModel):
    success: bool
    api_keys: list = []
    message: Optional[str] = None

@router.post("/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    """Register a new user"""
    result = AuthService.register(req.email, req.username, req.password)
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return AuthResponse(**result)

@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    """Login existing user"""
    result = AuthService.login(req.email, req.password)
    if not result["success"]:
        raise HTTPException(status_code=401, detail=result["error"])
    return AuthResponse(**result)

@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest,
    user: dict = Depends(get_current_user_from_jwt),
):
    """Change user password"""
    result = AuthService.change_password(
        user["user_id"], req.old_password, req.new_password
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

@router.get("/api-keys")
async def list_api_keys(user: dict = Depends(get_current_user_from_jwt)):
    """List all API keys for current user"""
    return AuthService.list_api_keys(user["user_id"])

@router.post("/revoke-key")
async def revoke_api_key(
    req: RevokeKeyRequest,
    user: dict = Depends(get_current_user_from_jwt),
):
    """Revoke an API key"""
    result = AuthService.revoke_api_key(req.api_key_id, user["user_id"])
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
