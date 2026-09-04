import json
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_current_user
from src.services.spell_grammar_service import SpellGrammarService

router = APIRouter(prefix="/v1", tags=["Spell & Grammar Check"])


class CheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                      json_schema_extra={"example": "This is a testt sentence with error."})
    type: str = Field("full", pattern="^(spell|grammar|full)$",
                      description="spell, grammar, or full")
    model: str = Field(
        "finetune", pattern="^(base|finetune)$", description="base or finetune"
    )
    language: str = Field("en", pattern="^(en|vi)$", description="en or vi")


class CorrectionDetail(BaseModel):
    original: str
    corrected: str
    type: str
    position: Optional[int] = None
    confidence: Optional[float] = None
    suggestion: Optional[str] = None


class CheckResponse(BaseModel):
    success: bool
    request_id: str
    original_text: str
    corrected_text: str
    corrections: List[CorrectionDetail] = []
    correction_count: int = 0
    processing_time_ms: int = 0
    cost_usd: float = 0.0
    model_used: str = ""
    error: Optional[str] = None


@router.post("/check", response_model=CheckResponse)
async def check_text(
    request: Request,
    check_req: CheckRequest,
    user: dict = Depends(get_current_user)
):
    """Check spelling and grammar with model selection"""
    service = SpellGrammarService()

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    result = await service.check(
        user_id=user["user_id"],
        api_key_id=user["api_key_id"],
        text=check_req.text,
        check_type=check_req.type,
        model=check_req.model,
        client_ip=client_ip,
        user_agent=user_agent
    )

    if not result.get("success"):
        status_code = 429 if "limit exceeded" in str(result.get("error", "")).lower() else 500
        raise HTTPException(
            status_code=status_code,
            detail=result.get("error", "Internal error"),
        )

    return CheckResponse(**result)


@router.post("/check/stream")
async def check_text_stream(
    request: Request,
    check_req: CheckRequest,
    user: dict = Depends(get_current_user)
):
    """Stream spell and grammar correction via Server-Sent Events (SSE)"""
    service = SpellGrammarService()
    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")

    async def event_generator():
        result = await service.check(
            user_id=user["user_id"],
            api_key_id=user["api_key_id"],
            text=check_req.text,
            check_type=check_req.type,
            model=check_req.model,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        # Yield metadata event
        yield f"event: metadata\ndata: {json.dumps({'request_id': result.get('request_id'), 'model_used': result.get('model_used')})}\n\n"

        # Yield chunks of corrected text
        corrected_text = result.get("corrected_text", check_req.text)
        chunk_size = 10
        for i in range(0, len(corrected_text), chunk_size):
            chunk = corrected_text[i:i + chunk_size]
            yield f"event: chunk\ndata: {json.dumps({'delta': chunk})}\n\n"

        # Yield final completion event
        yield f"event: complete\ndata: {json.dumps(result)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/spell", response_model=CheckResponse)
async def check_spell_only(
    request: Request,
    check_req: CheckRequest,
    user: dict = Depends(get_current_user)
):
    """Only spell check"""
    check_req.type = "spell"
    return await check_text(request, check_req, user)


@router.post("/grammar", response_model=CheckResponse)
async def check_grammar_only(
    request: Request,
    check_req: CheckRequest,
    user: dict = Depends(get_current_user)
):
    """Only grammar check"""
    check_req.type = "grammar"
    return await check_text(request, check_req, user)


