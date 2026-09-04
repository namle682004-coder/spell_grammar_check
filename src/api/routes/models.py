
from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_current_user
from src.services.inference import get_inference_service
from src.services.models import get_model_service

router = APIRouter(prefix="/v1/models", tags=["Model Management"])

@router.get("/")
async def list_models():
    """List all available models"""
    model_service = get_model_service()
    return model_service.list_models()

@router.get("/comparison")
async def compare_models():
    """Compare all models"""
    model_service = get_model_service()
    return model_service.get_model_comparison()

@router.get("/{model_name}")
async def get_model_info(
    model_name: str,
    user: dict = Depends(get_current_user)
):
    """Get detailed info for a specific model"""
    model_service = get_model_service()
    result = model_service.get_model_stats(model_name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.post("/{model_name}/load")
async def load_model(
    model_name: str,
    user: dict = Depends(get_current_user)
):
    """Load a model into memory"""
    # Check if user has permission (only admin)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    inference_service = get_inference_service()
    result = inference_service.load_model(model_name)

    return {
        "success": True,
        "model": result,
        "message": f"Model {model_name} loaded successfully"
    }

@router.get("/{model_name}/status")
async def get_model_status(
    model_name: str,
    user: dict = Depends(get_current_user)
):
    """Get model load status"""
    inference_service = get_inference_service()
    info = inference_service.get_model_info()

    for model in info.get("models", []):
        if model["name"] == model_name:
            return {
                "model": model_name,
                "is_loaded": model.get("is_loaded", False),
                "current_model": info.get("current_model")
            }

    raise HTTPException(status_code=404, detail="Model not found")
