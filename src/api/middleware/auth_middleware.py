import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.env import get_api_key

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({
    "/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/docs/")


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("X-API-Key")


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={
            "error": "unauthorized",
            "detail": "Invalid or missing API key. Use Authorize in /docs or send X-API-Key / Bearer header.",
        },
    )


def _misconfigured() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "configuration_error",
            "detail": "API_KEY is not set on the server. Add API_KEY to your .env file.",
        },
    )


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self._api_key = api_key if api_key is not None else get_api_key()

    async def dispatch(self, request: Request, call_next):
        if (
            request.method == "OPTIONS"
            or _is_public(request.url.path)
        ):
            return await call_next(request)

        if not self._api_key:
            return _misconfigured()

        token = _extract_token(request)
        if not token or not secrets.compare_digest(token, self._api_key):
            return _unauthorized()

        return await call_next(request)


def init_middleware(app):
    app.add_middleware(AuthMiddleware)
