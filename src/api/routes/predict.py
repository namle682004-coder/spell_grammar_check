from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.services.inference import get_inference_service

router = APIRouter(prefix="/v1/predict", tags=["Local Model Prediction"])

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    model: str = Field("finetune", description="base, finetune")

class PredictResponse(BaseModel):
    success: bool
    original_text: str
    corrected_text: str
    errors: List[dict] = []
    correction_count: int = 0
    processing_time_ms: int = 0
    model_used: str = ""
    error: Optional[str] = None

@router.post("/", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    user: dict = Depends(get_current_user)
):
    """Run local model prediction"""
    service = get_inference_service()

    result = service.predict(req.text, req.model)

    return PredictResponse(**result)

@router.post("/batch")
async def batch_predict(
    texts: List[str],
    model: str = "finetune",
    user: dict = Depends(get_current_user)
):
    """Batch prediction for multiple texts"""
    service = get_inference_service()

    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="Max 100 texts per batch")

    results = service.batch_predict(texts, model)

    return {
        "success": True,
        "count": len(results),
        "results": results
    }
