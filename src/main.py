import os

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.errors import register_exception_handlers
from src.api.middleware.logging_middleware import init_middleware as init_logging_middleware
from src.api.routes import (
    api_keys,
    auth,
    check,
    corrections,
    health,
    models,
    predict,
    quota,
    stats,
)

app = FastAPI(
    title="Spell & Grammar Check API",
    description="API for checking spelling and grammar errors",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)
init_logging_middleware(app)

app.include_router(auth.router)
app.include_router(check.router)
app.include_router(stats.router)
app.include_router(quota.router)
app.include_router(api_keys.router)
app.include_router(corrections.router)
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)


@app.get("/")
async def root():
    return {
        "message": "Spell & Grammar Check API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8080"))
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )
