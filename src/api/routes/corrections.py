
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List
from src.api.dependencies import get_current_user
from src.storage.database import get_db_manager
from src.storage.repositories import RequestRepository, CorrectionRepository
from pydantic import BaseModel, Field

router = APIRouter(prefix="/v1/corrections", tags=["Correction History"])

class CorrectionHistoryResponse(BaseModel):
    id: str
    request_id: str
    original_text: str
    corrected_text: str
    correction_type: str
    created_at: str

@router.get("/history")
async def get_correction_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    correction_type: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """Get correction history for current user"""
    db = get_db_manager()
    
    with db.get_session() as session:
        request_repo = RequestRepository(session)
        correction_repo = CorrectionRepository(session)
        
        # Get user's requests (status = "success")
        all_requests = request_repo.get_all(user_id=user["user_id"])
        requests = [r for r in all_requests if r.status == "success"]
        request_ids = [r.id for r in requests]
        
        if not request_ids:
            return {"corrections": [], "total": 0}
        
        # Get corrections for those requests
        corrections = []
        for req_id in request_ids:
            corrs = correction_repo.get_by_request(req_id)
            corrections.extend(corrs)
        
        # Filter by type if specified
        if correction_type:
            corrections = [c for c in corrections if c.correction_type == correction_type]
        
        # Paginate
        total = len(corrections)
        corrections = corrections[offset:offset + limit]
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "corrections": [
                {
                    "id": c.id,
                    "request_id": c.request_id,
                    "original_text": c.original_text[:200],
                    "corrected_text": c.corrected_text[:200],
                    "correction_type": c.correction_type,
                    "confidence": c.confidence,
                    "created_at": c.created_at.isoformat() if c.created_at else None
                }
                for c in corrections
            ]
        }

@router.get("/stats")
async def get_correction_stats(
    days: int = 30,
    user: dict = Depends(get_current_user)
):
    """Get correction statistics"""
    db = get_db_manager()
    
    with db.get_session() as session:
        from sqlalchemy import func
        from src.storage.models.correction_detail import CorrectionDetail
        from src.storage.models.spell_grammar_request import SpellGrammarRequest
        
        # Count by correction type
        stats = session.query(
            CorrectionDetail.correction_type,
            func.count(CorrectionDetail.id).label('count')
        ).join(
            SpellGrammarRequest,
            CorrectionDetail.request_id == SpellGrammarRequest.id
        ).filter(
            SpellGrammarRequest.user_id == user["user_id"]
        ).group_by(
            CorrectionDetail.correction_type
        ).all()
        
        return {
            "total_corrections": sum(s.count for s in stats),
            "breakdown": {
                s.correction_type: s.count
                for s in stats
            },
            "period_days": days
        }
