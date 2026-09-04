import hashlib
import json
import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.storage.context import RequestContext, reset_request_context, set_request_context
from src.storage.repository import get_usage_repository

logger = logging.getLogger(__name__)

SKIP_PATHS = frozenset({
    "/v1/health",
    "/docs",
    "/redoc",
    "/openapi.json",
})


def _client_id(request: Request) -> str:
    explicit = request.headers.get("X-Client-Id")
    if explicit:
        return explicit.strip()[:64]

    api_key = request.headers.get("X-API-Key")
    if not api_key:
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            api_key = auth[7:].strip()

    if api_key:
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "anonymous"


def _parse_error_code(response: Response) -> str | None:
    if response.status_code < 400:
        return None
    try:
        body = json.loads(response.body.decode())
        return body.get("error")
    except Exception:
        return None


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method == "OPTIONS" or request.url.path in SKIP_PATHS:
            return await call_next(request)

        ctx = RequestContext(
            request_id=uuid.uuid4().hex,
            client_id=_client_id(request),
            session_id=request.headers.get("X-Session-Id"),
        )
        set_request_context(ctx)
        request.state.usage_context = ctx

        start = time.time()
        response: Response = await call_next(request)
        latency_ms = (time.time() - start) * 1000

        repo = get_usage_repository()
        error_code = _parse_error_code(response) or ctx.error_code

        repo.log_api_request(
            request_id=ctx.request_id,
            client_id=ctx.client_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            model_type=ctx.model_type,
            input_chars=ctx.input_chars,
            output_chars=ctx.output_chars,
            error_code=error_code,
        )

        if ctx.prompt_tokens is not None:
            repo.log_llm_usage(ctx=ctx, endpoint=request.url.path)

        if response.status_code >= 400 and error_code:
            try:
                detail = json.loads(response.body.decode()).get("detail", "")
            except Exception:
                detail = ""
            repo.log_error(
                request_id=ctx.request_id,
                client_id=ctx.client_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                error_code=error_code,
                detail=str(detail),
            )

        response.headers["X-Request-Id"] = ctx.request_id
        reset_request_context()
        return response


def init_middleware(app):
    app.add_middleware(UsageTrackingMiddleware)
