import time
from typing import Dict, Any, List, Optional
import torch
from src.inference import predictor
from src.inference.predictor import BASE_MODEL, FINETUNE_MODEL
from src.api.middleware.errors import ApiError, ModelLoadError

class InferenceService:
    """Service for local model inference"""
    
    def __init__(self):
        self.current_model = None
        self.model_cache = {}
    
    def load_model(self, model_name: str = "finetune") -> str:
        """Load model by name"""
        model_path = self._resolve_model_path(model_name)
        
        try:
            predictor.load_model(model_path)
            self.current_model = model_name
            return model_name
        except ApiError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model {model_name}", detail=str(exc))
    
    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve model name to path"""
        model_map = {
            "base": BASE_MODEL,
            "finetune": FINETUNE_MODEL,
            "fast": BASE_MODEL,
            "accurate": FINETUNE_MODEL
        }
        return model_map.get(model_name, FINETUNE_MODEL)
    
    def predict(self, text: str, model_name: str = "finetune") -> Dict[str, Any]:
        """Run inference on text"""
        start_time = time.time()
        
        # Load model if needed
        if self.current_model != model_name:
            self.load_model(model_name)
        
        try:
            from src.inference.predictor import generate_with_usage

            gen = generate_with_usage(text, model_type=model_name)
            corrected = gen.output
            if "assistant" in corrected:
                corrected = corrected.split("assistant")[-1].strip()
            elif "Đoạn đã sửa:" in corrected:
                corrected = corrected.split("Đoạn đã sửa:")[-1].strip()

            processing_time = int((time.time() - start_time) * 1000)

            return {
                "success": True,
                "original_text": text,
                "corrected_text": corrected.strip() or text,
                "errors": [],
                "correction_count": 0,
                "processing_time_ms": processing_time,
                "model_used": model_name,
                "confidence": 0.85
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "original_text": text,
                "corrected_text": text,
                "errors": [],
                "correction_count": 0,
                "processing_time_ms": int((time.time() - start_time) * 1000),
                "model_used": model_name
            }
    
    def _format_errors(self, errors: List[Dict]) -> List[Dict]:
        """Format errors from predictor"""
        formatted = []
        for error in errors:
            formatted.append({
                "original": error.get("original", ""),
                "corrected": error.get("corrected", ""),
                "type": error.get("type", "unknown"),
                "position": error.get("position", 0),
                "confidence": error.get("confidence", 0.9),
                "suggestion": error.get("suggestion", "")
            })
        return formatted
    
    def batch_predict(self, texts: List[str], model_name: str = "finetune") -> List[Dict[str, Any]]:
        """Run inference on multiple texts"""
        results = []
        for text in texts:
            result = self.predict(text, model_name)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "models": [
                {
                    "name": "base",
                    "path": BASE_MODEL,
                    "description": "Base model without fine-tuning",
                    "type": "transformer"
                },
                {
                    "name": "finetune",
                    "path": FINETUNE_MODEL,
                    "description": "Fine-tuned model for spell/grammar",
                    "type": "transformer"
                }
            ],
            "current_model": self.current_model,
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        }

# Singleton instance
_inference_service: Optional[InferenceService] = None

def get_inference_service() -> InferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service