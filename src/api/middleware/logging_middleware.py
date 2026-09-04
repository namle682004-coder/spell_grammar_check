import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response: Response = await call_next(request)
        process_time = (time.time() - start) * 1000
        logging.info(
            f"{request.method} {request.url.path} completed_in={process_time:.2f}ms status_code={response.status_code}"
        )
        response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"
        return response


def init_middleware(app):
    app.add_middleware(RequestLoggingMiddleware)
