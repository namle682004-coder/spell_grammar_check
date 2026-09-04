
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from src.services.llm_service import LLMService
from src.storage.database import get_db_manager
from src.storage.repositories import (
    UserRepository,
    ApiKeyRepository,
    RequestRepository,
    CorrectionRepository,
    UsageRepository
)


class SpellGrammarService:
    def __init__(self):
        self.llm_service = LLMService()
    
    def _get_or_create_quota(self, usage_repo: UsageRepository, user_id: str):
        """Lấy quota hiện tại hoặc tạo mới nếu chưa có"""
        quota = usage_repo.get_quota_by_user_id(user_id)
        
        if not quota:
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_end = (month_start + timedelta(days=32)).replace(day=1)
            
            quota = usage_repo.create_quota(
                user_id=user_id,
                period="monthly",
                period_start=month_start,
                period_end=month_end,
                request_limit=100,
                token_limit=10000,
                cost_limit_usd=1.0,
                requests_used=0,
                tokens_used=0,
                cost_used_usd=0.0,
                requests_last_minute=0,
                requests_last_hour=0,
                is_exceeded=False,
                exceeded_reason=""
            )
        
        return quota
    
    def _check_quota(self, quota) -> tuple:
        """Kiểm tra quota còn không, trả về (allowed, reason)"""
        if quota.requests_used >= quota.request_limit:
            return False, f"Monthly request limit exceeded: {quota.requests_used}/{quota.request_limit}"
        
        if quota.tokens_used >= quota.token_limit:
            return False, f"Monthly token limit exceeded: {quota.tokens_used}/{quota.token_limit}"
        
        if quota.cost_used_usd >= quota.cost_limit_usd:
            return False, f"Monthly budget exceeded: ${quota.cost_used_usd:.2f}/${quota.cost_limit_usd:.2f}"
        
        return True, "OK"
    
    def _consume_quota(self, usage_repo: UsageRepository, quota, tokens: int, cost: float):
        """Trừ quota sau khi xử lý thành công"""
        usage_repo.update_quota(
            quota.id,
            requests_used=quota.requests_used + 1,
            tokens_used=quota.tokens_used + tokens,
            cost_used_usd=quota.cost_used_usd + cost
        )

    def _persist_corrections(
        self,
        correction_repo: CorrectionRepository,
        request_db_id: str,
        original_text: str,
        corrections: list,
    ) -> None:
        for item in corrections:
            correction_type = item.get("type", "grammar")
            if correction_type not in ("spelling", "grammar", "punctuation", "style", "typo"):
                correction_type = "grammar"
            correction_repo.create(
                request_id=request_db_id,
                correction_type=correction_type,
                original_text=item.get("original", ""),
                corrected_text=item.get("corrected", ""),
                start_char=item.get("position", 0),
                end_char=item.get("position", 0) + len(item.get("original", "")),
                confidence=item.get("confidence", 0.0),
                suggestion=item.get("suggestion", ""),
            )

    async def check(
        self,
        user_id: str,
        api_key_id: str,
        text: str,
        check_type: str = "full",
        model: str = "finetune",
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:

        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        cost_usd = 0.001

        db = get_db_manager()

        # Step 1: Mở DB session 1 để kiểm tra Quota & tạo Record Request (status=processing)
        with db.get_session() as session:
            request_repo = RequestRepository(session)
            usage_repo = UsageRepository(session)
            
            quota = self._get_or_create_quota(usage_repo, user_id)
            
            allowed, reason = self._check_quota(quota)
            if not allowed:
                return {
                    "success": False,
                    "error": reason,
                    "quota_remaining": {
                        "requests": 0,
                        "tokens": 0,
                        "budget_usd": 0
                    }
                }
            
            request = request_repo.create(
                request_id=request_id,
                user_id=user_id,
                api_key_id=api_key_id,
                request_type=check_type,
                input_text=text,
                input_chars=len(text),
                input_tokens=0,
                status="processing",
                client_ip=client_ip,
                user_agent=user_agent,
                model_used=model
            )
            request_db_id = request.id

        # Step 2: Check cache hoặc gọi LLM Service
        cached_result = global_cache.get(text, model)
        if cached_result:
            result = cached_result
        else:
            try:
                result = await self.llm_service.check_spell_grammar(text, model)
                if result.get("success"):
                    global_cache.set(text, result, model)
            except Exception as e:
                result = {
                    "success": False,
                    "error": str(e),
                }

        processing_time = int((time.time() - start_time) * 1000)
        tokens_used = result.get("total_tokens") or (
            (result.get("prompt_tokens") or 0) + (result.get("completion_tokens") or 0)
        ) or len(text)

        # Step 3: Mở DB session 2 để cập nhật kết quả, trừ quota & lưu corrections
        with db.get_session() as session:
            request_repo = RequestRepository(session)
            api_key_repo = ApiKeyRepository(session)
            usage_repo = UsageRepository(session)
            correction_repo = CorrectionRepository(session)

            quota = self._get_or_create_quota(usage_repo, user_id)

            if result.get("success"):
                corrections = result.get("errors", [])
                request_repo.update(
                    request_db_id,
                    output_text=result["corrected_text"],
                    output_chars=len(result["corrected_text"]),
                    total_corrections=result["correction_count"],
                    model_used=result.get("model_used", model),
                    processing_time_ms=processing_time,
                    input_tokens=result.get("prompt_tokens", tokens_used),
                    output_tokens=result.get("completion_tokens", 0),
                    cost_usd=cost_usd,
                    status="success"
                )

                if corrections:
                    self._persist_corrections(
                        correction_repo, request_db_id, text, corrections
                    )

                self._consume_quota(usage_repo, quota, tokens_used, cost_usd)
                api_key_repo.increment_usage(api_key_id, tokens=tokens_used)

                remaining_requests = quota.request_limit - (quota.requests_used + 1)
                remaining_tokens = quota.token_limit - (quota.tokens_used + tokens_used)
                remaining_budget = quota.cost_limit_usd - (quota.cost_used_usd + cost_usd)

                return {
                    "success": True,
                    "request_id": request_id,
                    "original_text": text,
                    "corrected_text": result["corrected_text"],
                    "corrections": corrections,
                    "correction_count": result["correction_count"],
                    "processing_time_ms": processing_time,
                    "cost_usd": cost_usd,
                    "model_used": result.get("model_used", model),
                    "quota_remaining": {
                        "requests": max(0, remaining_requests),
                        "tokens": max(0, remaining_tokens),
                        "budget_usd": round(max(0, remaining_budget), 4)
                    }
                }
            else:
                request_repo.update(
                    request_db_id,
                    status="failed",
                    error_message=result.get("error", "Model error")
                )
                return {
                    "success": False,
                    "request_id": request_id,
                    "error": result.get("error", "Failed to process"),
                    "original_text": text,
                    "corrected_text": text,
                    "corrections": [],
                    "correction_count": 0,
                    "processing_time_ms": processing_time,
                    "cost_usd": 0,
                    "model_used": model
                }

