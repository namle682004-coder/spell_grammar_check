"""API-layer exceptions mapped to HTTP responses by global handlers."""


class ApiError(Exception):
    status_code: int = 500
    error_code: str = "service_error"

    def __init__(self, message: str, *, detail: str | None = None):
        self.message = message
        self.detail = detail or message
        super().__init__(self.message)


class InferenceError(ApiError):
    status_code = 500
    error_code = "inference_error"


class ModelLoadError(ApiError):
    status_code = 503
    error_code = "model_load_error"


class InvalidModelTypeError(ApiError):
    status_code = 400
    error_code = "invalid_model_type"
