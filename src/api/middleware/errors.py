from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import (
    DatabaseError,
    DataError,
    IntegrityError,
    OperationalError,
    ProgrammingError,
)
from starlette.exceptions import HTTPException as StarletteHTTPException

try:
    import httpx
except ImportError:
    httpx = None

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)

# ============== ĐỊNH NGHĨA CÁC EXCEPTION CLASS ==============


class ApiError(Exception):
    """Base API exception"""

    def __init__(self, error_code: str, detail: str = None, status_code: int = 500):
        self.error_code = error_code
        self.detail = detail
        self.status_code = status_code
        super().__init__(error_code)


class ModelLoadError(ApiError):
    """Exception raised when model fails to load"""

    def __init__(self, message: str = "Failed to load model", detail: str = None):
        super().__init__(
            error_code="model_load_error",
            detail=detail or message,
            status_code=500
        )


class QuotaExceededError(ApiError):
    """Exception raised when quota is exceeded"""

    def __init__(self, message: str = "Quota exceeded", detail: str = None):
        super().__init__(
            error_code="quota_exceeded",
            detail=detail or message,
            status_code=429
        )


class InvalidAPIKeyError(ApiError):
    """Exception raised when API key is invalid"""

    def __init__(self, message: str = "Invalid API key", detail: str = None):
        super().__init__(
            error_code="invalid_api_key",
            detail=detail or message,
            status_code=401
        )


class RateLimitError(ApiError):
    """Exception raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded", detail: str = None):
        super().__init__(
            error_code="rate_limit_exceeded",
            detail=detail or message,
            status_code=429
        )


# ============== PHẦN CŨ CỦA BẠN (GIỮ NGUYÊN) ==============
_STATUS_TO_ERROR = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    422: "validation_error",
    500: "internal_server_error",
    503: "service_unavailable",
}


def _error_payload(error: str, detail: object, **extra) -> dict:
    payload = {"error": error, "detail": detail}
    payload.update(extra)
    return payload


def _status_error_code(status_code: int) -> str:
    return _STATUS_TO_ERROR.get(status_code, "http_error")


def _normalize_detail(detail: object) -> object:
    if isinstance(detail, str):
        return detail
    return detail


def register_exception_handlers(app) -> None:
    @app.exception_handler(ApiError)
    async def api_error_handler(request: Request, exc: ApiError):
        logger.warning(
            "%s %s api_error=%s detail=%s",
            request.method,
            request.url.path,
            exc.error_code,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(exc.error_code, exc.detail),
        )

    @app.exception_handler(HTTPException)
    async def fastapi_http_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                _status_error_code(exc.status_code),
                _normalize_detail(exc.detail),
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def starlette_http_handler(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_payload(
                _status_error_code(exc.status_code),
                _normalize_detail(exc.detail),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_error_payload("validation_error", exc.errors()),
        )

    @app.exception_handler(EnvironmentError)
    async def environment_error_handler(request: Request, exc: EnvironmentError):
        logger.error("%s %s environment_error=%s",
                     request.method, request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content=_error_payload("configuration_error", str(exc)),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(
            status_code=400,
            content=_error_payload("bad_request", str(exc)),
        )

    @app.exception_handler(KeyError)
    async def key_error_handler(request: Request, exc: KeyError):
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                "missing_key", f"Missing required key: {exc}"),
        )

    @app.exception_handler(TypeError)
    async def type_error_handler(request: Request, exc: TypeError):
        logger.exception("%s %s type_error", request.method, request.url.path)
        return JSONResponse(
            status_code=400,
            content=_error_payload("invalid_type", str(exc)),
        )

    @app.exception_handler(AttributeError)
    async def attribute_error_handler(request: Request, exc: AttributeError):
        logger.exception("%s %s attribute_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload("service_error", str(exc)),
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.exception("%s %s file_not_found",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload("resource_not_found", str(exc)),
        )

    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        logger.exception("%s %s permission_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=403,
            content=_error_payload("permission_denied", str(exc)),
        )

    @app.exception_handler(NotImplementedError)
    async def not_implemented_handler(request: Request, exc: NotImplementedError):
        logger.exception("%s %s not_implemented",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=501,
            content=_error_payload("not_implemented", str(exc)),
        )

    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        logger.exception("%s %s timeout_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=504,
            content=_error_payload(
                "timeout_error",
                "The request timed out. Please retry later or check your network connection.",
            ),
        )

    @app.exception_handler(ConnectionError)
    async def connection_error_handler(request: Request, exc: ConnectionError):
        logger.exception("%s %s connection_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=503,
            content=_error_payload(
                "connection_error",
                "A network connection error occurred. Please check your network or external service availability.",
            ),
        )

    @app.exception_handler(OSError)
    async def os_error_handler(request: Request, exc: OSError):
        logger.exception("%s %s os_error", request.method, request.url.path)
        return JSONResponse(
            status_code=503,
            content=_error_payload("service_unavailable", str(exc)),
        )

    @app.exception_handler(OperationalError)
    async def sqlalchemy_operational_error_handler(request: Request, exc: OperationalError):
        logger.exception("%s %s database_operational_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=503,
            content=_error_payload(
                "database_connection_error",
                "Cannot connect to the database. Please check that PostgreSQL is running and DATABASE_URL is correct.",
            ),
        )

    @app.exception_handler(DatabaseError)
    async def sqlalchemy_database_error_handler(request: Request, exc: DatabaseError):
        logger.exception("%s %s database_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=503,
            content=_error_payload(
                "database_error",
                "A database error occurred. Check the service status and database configuration.",
            ),
        )

    @app.exception_handler(IntegrityError)
    async def sqlalchemy_integrity_error_handler(request: Request, exc: IntegrityError):
        logger.exception("%s %s database_integrity_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=409,
            content=_error_payload(
                "database_integrity_error",
                "A database constraint was violated. Check the submitted data.",
            ),
        )

    @app.exception_handler(DataError)
    async def sqlalchemy_data_error_handler(request: Request, exc: DataError):
        logger.exception("%s %s database_data_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                "database_data_error",
                "Invalid data was passed to the database. Verify input formats and types.",
            ),
        )

    @app.exception_handler(ProgrammingError)
    async def sqlalchemy_programming_error_handler(request: Request, exc: ProgrammingError):
        logger.exception("%s %s database_programming_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "database_programming_error",
                "A database programming error occurred. Check database schema and queries.",
            ),
        )

    if httpx is not None:
        @app.exception_handler(httpx.RequestError)
        async def httpx_request_error_handler(request: Request, exc: "httpx.RequestError"):
            logger.exception("%s %s httpx_request_error",
                             request.method, request.url.path)
            return JSONResponse(
                status_code=503,
                content=_error_payload(
                    "external_service_error",
                    "An external HTTP request failed. Please check the external service or network connectivity.",
                ),
            )

        @app.exception_handler(httpx.TimeoutException)
        async def httpx_timeout_error_handler(request: Request, exc: "httpx.TimeoutException"):
            logger.exception("%s %s httpx_timeout",
                             request.method, request.url.path)
            return JSONResponse(
                status_code=504,
                content=_error_payload(
                    "external_timeout",
                    "An external request timed out. Please retry later.",
                ),
            )

    if requests is not None:
        @app.exception_handler(requests.exceptions.RequestException)
        async def requests_request_exception_handler(request: Request, exc: "requests.exceptions.RequestException"):
            logger.exception("%s %s requests_exception",
                             request.method, request.url.path)
            return JSONResponse(
                status_code=503,
                content=_error_payload(
                    "external_service_error",
                    "An external HTTP request failed. Please check the external service or network connectivity.",
                ),
            )

    @app.exception_handler(ImportError)
    async def import_error_handler(request: Request, exc: ImportError):
        logger.exception("%s %s import_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "import_error",
                "A required dependency failed to import. Check server dependencies and environment.",
            ),
        )

    try:
        import torch

        if (
            hasattr(torch, "cuda")
            and hasattr(torch.cuda, "OutOfMemoryError")
            and isinstance(torch.cuda.OutOfMemoryError, type)
            and issubclass(torch.cuda.OutOfMemoryError, Exception)
        ):
            @app.exception_handler(torch.cuda.OutOfMemoryError)
            async def cuda_oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
                logger.error("%s %s cuda_oom", request.method, request.url.path)
                return JSONResponse(
                    status_code=503,
                    content=_error_payload(
                        "gpu_out_of_memory",
                        "GPU ran out of memory. Try a smaller batch or unload the current model.",
                    ),
                )
    except Exception:
        pass

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.exception("%s %s runtime_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload("service_error", str(exc)),
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.exception("%s %s unhandled_error",
                         request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "internal_server_error",
                "An unexpected error occurred. Check server logs for details.",
            ),
        )
