from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, between
from src.storage.models.spell_grammar_request import SpellGrammarRequest, RequestType, RequestStatus
from src.storage.repositories.base_repo import BaseRepository
import uuid

class RequestRepository(BaseRepository[SpellGrammarRequest]):
    def __init__(self, session: Session):
        super().__init__(SpellGrammarRequest, session)
    
    def create_request(self, user_id: str, request_type: RequestType, input_text: str, **kwargs) -> SpellGrammarRequest:
        """Tạo request mới"""
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        return self.create(
            request_id=request_id,
            user_id=user_id,
            request_type=request_type,
            input_text=input_text,
            input_chars=len(input_text),
            **kwargs
        )
    
    def update_result(self, request_id: str, output_text: str, corrections_count: int, **kwargs) -> Optional[SpellGrammarRequest]:
        """Cập nhật kết quả cho request"""
        return self.update(
            request_id,
            output_text=output_text,
            output_chars=len(output_text),
            total_corrections=corrections_count,
            status=RequestStatus.SUCCESS,
            **kwargs
        )
    
    def mark_failed(self, request_id: str, error_message: str, status_code: int = 500) -> Optional[SpellGrammarRequest]:
        """Đánh dấu request bị lỗi"""
        return self.update(
            request_id,
            status=RequestStatus.FAILED,
            error_message=error_message[:500],
            status_code=status_code
        )
    
    def get_user_requests(self, user_id: str, days: int = 30, skip: int = 0, limit: int = 100) -> List[SpellGrammarRequest]:
        """Lấy lịch sử requests của user"""
        since_date = datetime.utcnow() - timedelta(days=days)
        return self.session.query(SpellGrammarRequest).filter(
            and_(
                SpellGrammarRequest.user_id == user_id,
                SpellGrammarRequest.created_at >= since_date
            )
        ).order_by(SpellGrammarRequest.created_at.desc()).offset(skip).limit(limit).all()
    
    def get_stats_by_period(self, user_id: str, start_date: datetime, end_date: datetime) -> dict:
        """Lấy thống kê requests theo khoảng thời gian"""
        stats = self.session.query(
            SpellGrammarRequest.request_type,
            func.count(SpellGrammarRequest.id).label('total'),
            func.avg(SpellGrammarRequest.processing_time_ms).label('avg_processing_time'),
            func.sum(SpellGrammarRequest.total_corrections).label('total_corrections'),
            func.sum(SpellGrammarRequest.input_tokens + SpellGrammarRequest.output_tokens).label('total_tokens'),
            func.sum(SpellGrammarRequest.cost_usd).label('total_cost')
        ).filter(
            and_(
                SpellGrammarRequest.user_id == user_id,
                between(SpellGrammarRequest.created_at, start_date, end_date),
                SpellGrammarRequest.status == RequestStatus.SUCCESS
            )
        ).group_by(SpellGrammarRequest.request_type).all()
        
        result = {}
        for stat in stats:
            result[stat.request_type.value] = {
                "total_requests": stat.total,
                "avg_processing_time_ms": round(stat.avg_processing_time, 2) if stat.avg_processing_time else 0,
                "total_corrections": stat.total_corrections,
                "total_tokens": stat.total_tokens or 0,
                "total_cost_usd": round(stat.total_cost or 0, 6)
            }
        
        return result
    
    def get_daily_usage(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """Lấy usage theo ngày"""
        since_date = datetime.utcnow() - timedelta(days=days)
        
        results = self.session.query(
            func.date(SpellGrammarRequest.created_at).label('date'),
            func.count(SpellGrammarRequest.id).label('requests'),
            func.sum(SpellGrammarRequest.total_corrections).label('corrections'),
            func.sum(SpellGrammarRequest.input_tokens + SpellGrammarRequest.output_tokens).label('tokens'),
            func.sum(SpellGrammarRequest.cost_usd).label('cost')
        ).filter(
            and_(
                SpellGrammarRequest.user_id == user_id,
                SpellGrammarRequest.created_at >= since_date,
                SpellGrammarRequest.status == RequestStatus.SUCCESS
            )
        ).group_by(func.date(SpellGrammarRequest.created_at)).order_by('date').all()
        
        return [
            {
                "date": str(r.date),
                "requests": r.requests,
                "corrections": r.corrections,
                "tokens": r.tokens or 0,
                "cost_usd": round(r.cost or 0, 6)
            }
            for r in results
        ]
    
    def get_top_corrections(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Lấy các loại lỗi phổ biến nhất của user"""
        from src.storage.models.correction_detail import CorrectionDetail, CorrectionType
        
        results = self.session.query(
            CorrectionDetail.correction_type,
            func.count(CorrectionDetail.id).label('count')
        ).join(
            SpellGrammarRequest,
            CorrectionDetail.request_id == SpellGrammarRequest.id
        ).filter(
            SpellGrammarRequest.user_id == user_id
        ).group_by(
            CorrectionDetail.correction_type
        ).order_by(
            func.count(CorrectionDetail.id).desc()
        ).limit(limit).all()
        
        return [
            {"type": r.correction_type.value, "count": r.count}
            for r in results
        ]