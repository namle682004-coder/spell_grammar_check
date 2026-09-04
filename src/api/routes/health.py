from datetime import datetime

import httpx
import psutil
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from src.inference.vllm_client import get_vllm_base_url

router = APIRouter(tags=["Health"])


@router.get("/health")
@router.get("/v1/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


async def _check_db_healthy() -> tuple[bool, str]:
    try:
        from sqlalchemy import text

        from src.storage.database import get_db_manager
        db = get_db_manager()
        with db.get_session() as session:
            session.execute(text("SELECT 1"))
            return True, "healthy"
    except Exception as e:
        return False, f"unhealthy: {str(e)}"


async def _check_vllm_healthy() -> tuple[bool, str]:
    vllm_url = get_vllm_base_url()
    if not vllm_url:
        return True, "disabled (local fallback)"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{vllm_url.rstrip('/')}/health")
            if resp.status_code == 200:
                return True, "healthy"
            return False, f"unhealthy status {resp.status_code}"
    except Exception as e:
        return False, f"unhealthy: {str(e)}"


@router.get("/health/detailed")
@router.get("/v1/health/detailed")
async def detailed_health():
    """Detailed health check with system and dependency info"""
    db_ok, db_msg = await _check_db_healthy()
    vllm_ok, vllm_msg = await _check_vllm_healthy()

    is_healthy = db_ok and vllm_ok
    status_code = status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE

    content = {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": db_msg,
            "vllm_inference": vllm_msg,
            "api": "healthy",
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
        },
    }
    return JSONResponse(status_code=status_code, content=content)


@router.get("/ready")
async def readiness_check():
    """Readiness probe for k8s/load balancer"""
    db_ok, db_msg = await _check_db_healthy()
    if not db_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": f"Database {db_msg}"},
        )

    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Liveness probe for k8s"""
    return {"status": "alive"}