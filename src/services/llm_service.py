import difflib
import re
import time
from typing import Any, Dict, List

from src.inference.predictor import generate_with_usage
from src.inference.vllm_client import generate_via_vllm_async, get_vllm_base_url


class LLMService:
    def _extract_corrections(self, original_text: str, corrected_text: str) -> List[Dict]:
        errors = []
        orig_words = original_text.split()
        corr_words = corrected_text.split()

        if not orig_words or not corr_words:
            return errors

        # Compute character start positions for each word in original_text
        word_positions = []
        current_pos = 0
        for w in orig_words:
            pos = original_text.find(w, current_pos)
            if pos != -1:
                word_positions.append(pos)
                current_pos = pos + len(w)
            else:
                word_positions.append(current_pos)

        matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue

            orig_chunk = " ".join(orig_words[i1:i2])
            corr_chunk = " ".join(corr_words[j1:j2])
            pos = word_positions[i1] if i1 < len(word_positions) else len(original_text)

            err_type = "grammar" if tag in ("insert", "delete") or (i2 - i1) > 1 or (j2 - j1) > 1 else "spelling"
            suggestion = f"Should be '{corr_chunk}'" if corr_chunk else "Should be removed"

            errors.append({
                "original": orig_chunk,
                "corrected": corr_chunk,
                "type": err_type,
                "position": pos,
                "confidence": 0.9,
                "suggestion": suggestion,
            })

        return errors

    def _parse_corrected_output(self, text: str, output_text: str) -> str:
        corrected = output_text
        if "assistant" in output_text:
            corrected = output_text.split("assistant")[-1].strip()
        elif "Đoạn đã sửa:" in output_text:
            corrected = output_text.split("Đoạn đã sửa:")[-1].strip()
            if "assistant" in corrected:
                corrected = corrected.split("assistant")[-1].strip()

        if text in corrected:
            corrected = corrected.replace(text, "").strip()

        corrected = re.sub(r"^[\n\s]+", "", corrected)
        corrected = re.sub(r"[\n\s]+$", "", corrected)
        return corrected

    async def check_spell_grammar(self, text: str, model_name: str = "finetune") -> Dict[str, Any]:
        start_time = time.time()

        try:
            if get_vllm_base_url():
                vllm_result = await generate_via_vllm_async(text=text, model_name=model_name)
                result_output = vllm_result.output
                prompt_tokens = vllm_result.prompt_tokens
                completion_tokens = vllm_result.completion_tokens
                total_tokens = vllm_result.total_tokens
                model_used = vllm_result.model_name
            else:
                result = generate_with_usage(text=text, model_type=model_name)
                result_output = result.output
                prompt_tokens = result.prompt_tokens
                completion_tokens = result.completion_tokens
                total_tokens = result.total_tokens
                model_used = result.model_name

            corrected = self._parse_corrected_output(text, result_output)
            errors = self._extract_corrections(text, corrected)
            processing_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "corrected_text": corrected if corrected else text,
                "errors": errors,
                "correction_count": len(errors),
                "model_used": model_used,
                "processing_time_ms": processing_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_ms": processing_time,
            }
        except Exception as e:
            return {
                "success": False,
                "corrected_text": text,
                "errors": [],
                "correction_count": 0,
                "model_used": model_name,
                "error": str(e),
            }

