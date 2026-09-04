
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from src.api.middleware.errors import RateLimitError
from src.services.auth_service import AuthService

# Single primary security scheme for OpenAPI / Swagger UI
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key_hdr: str | None = Security(api_key_header),
):
    """Unified Auth dependency supporting both API Keys (sg_...) and JWT Tokens.

    Supports input via:
    - Authorization: Bearer <API_KEY_OR_JWT_TOKEN>
    - X-API-Key: <API_KEY>
    """
    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials.strip()
    elif api_key_hdr:
        token = api_key_hdr.strip()

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing authentication token or API key. Provide via 'Authorization: Bearer <token>' or 'X-API-Key: <key>'",
        )

    # 1. Try verifying as API Key (if starts with sg_)
    if token.startswith("sg_") or "_" in token:
        try:
            user_info = AuthService.verify_api_key(token)
            if user_info:
                return user_info
        except RateLimitError as exc:
            raise HTTPException(status_code=429, detail=exc.detail or str(exc))

    # 2. Try verifying as JWT Token
    jwt_user = AuthService.verify_jwt(token)
    if jwt_user:
        if "api_key_id" not in jwt_user:
            jwt_user["api_key_id"] = "jwt_session"
        return jwt_user

    # 3. Fallback: try verify_api_key in case API key doesn't start with sg_
    try:
        user_info = AuthService.verify_api_key(token)
        if user_info:
            return user_info
    except RateLimitError as exc:
        raise HTTPException(status_code=429, detail=exc.detail or str(exc))

    raise HTTPException(status_code=401, detail="Invalid or expired API Key or JWT token")


async def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key_hdr: str | None = Security(api_key_header),
):
    """Optional auth - returns None if not authenticated."""
    if not credentials and not api_key_hdr:
        return None
    try:
        return await get_current_user(credentials, api_key_hdr)
    except HTTPException:
        return None


async def get_current_user_from_jwt(
    credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    api_key_hdr: str | None = Security(api_key_header),
):
    """Alias for unified auth (supports JWT Bearer and API Key)."""
    return await get_current_user(credentials, api_key_hdr)

