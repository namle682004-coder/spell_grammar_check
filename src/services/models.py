import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelInfo:
    """Information about a model"""
    name: str
    path: str
    version: str
    size_mb: float
    accuracy: float
    speed_ms: int
    is_loaded: bool = False
    last_used: Optional[datetime] = None

class ModelService:
    """Service for managing multiple models"""
    
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self._load_model_config()
    
    def _load_model_config(self):
        """Load model configuration from file or environment"""
        # Default models
        self.models = {
            "base": ModelInfo(
                name="base",
                path=os.getenv("BASE_MODEL_PATH", "models/base"),
                version="1.0.0",
                size_mb=450,
                accuracy=0.85,
                speed_ms=50
            ),
            "finetune": ModelInfo(
                name="finetune",
                path=os.getenv("FINETUNE_MODEL_PATH", "models/finetuned"),
                version="2.0.0",
                size_mb=450,
                accuracy=0.94,
                speed_ms=80
            )
        }
        
        # Try to load from config file
        config_path = os.getenv("MODEL_CONFIG_PATH", "configs/models.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    for name, data in config.items():
                        if name in self.models:
                            self.models[name] = ModelInfo(**data)
            except Exception as e:
                print(f"Error loading model config: {e}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models"""
        return [
            {
                "name": name,
                "version": info.version,
                "size_mb": info.size_mb,
                "accuracy": info.accuracy,
                "speed_ms": info.speed_ms,
                "is_loaded": info.is_loaded,
                "description": self._get_model_description(name)
            }
            for name, info in self.models.items()
        ]
    
    def get_model(self, model_name: str) -> Optional[ModelInfo]:
        """Get model info by name"""
        return self.models.get(model_name)
    
    def get_default_model(self) -> str:
        """Get default model name"""
        return os.getenv("DEFAULT_MODEL", "finetune")
    
    def get_model_comparison(self) -> Dict[str, Any]:
        """Compare all models"""
        comparison = {
            "fastest": min(self.models.items(), key=lambda x: x[1].speed_ms)[0],
            "most_accurate": max(self.models.items(), key=lambda x: x[1].accuracy)[0],
            "smallest": min(self.models.items(), key=lambda x: x[1].size_mb)[0]
        }
        
        return {
            "comparison": comparison,
            "models": self.list_models(),
            "recommendation": self._get_recommendation()
        }
    
    def _get_model_description(self, model_name: str) -> str:
        """Get description for a model"""
        descriptions = {
            "base": "Base model without fine-tuning. Fast but less accurate.",
            "finetune": "Fine-tuned model for spell/grammar correction. Best accuracy."
        }
        return descriptions.get(model_name, "Unknown model")
    
    def _get_recommendation(self) -> str:
        """Get model recommendation based on use case"""
        return """
        - For production: Use 'finetune' for best accuracy
        - For testing/development: Use 'base' for faster iteration
        - For real-time applications: Consider 'base' for lower latency
        - For critical grammar checking: Use 'finetune'
        """
    
    def update_model_status(self, model_name: str, is_loaded: bool):
        """Update model load status"""
        if model_name in self.models:
            self.models[model_name].is_loaded = is_loaded
            if is_loaded:
                self.models[model_name].last_used = datetime.utcnow()
    
    def get_model_stats(self, model_name: str) -> Dict[str, Any]:
        """Get detailed statistics for a model"""
        model = self.models.get(model_name)
        if not model:
            return {"error": "Model not found"}
        
        return {
            "name": model.name,
            "version": model.version,
            "size_mb": model.size_mb,
            "accuracy": model.accuracy,
            "avg_speed_ms": model.speed_ms,
            "is_loaded": model.is_loaded,
            "last_used": model.last_used.isoformat() if model.last_used else None,
            "estimated_cost_per_1k_chars": self._estimate_cost(model_name)
        }
    
    def _estimate_cost(self, model_name: str) -> float:
        """Estimate cost per 1000 characters"""
        costs = {
            "base": 0.0001,  # $0.0001 per 1K chars
            "finetune": 0.0005  # $0.0005 per 1K chars
        }
        return costs.get(model_name, 0.0001)

# Singleton instance
_model_service: Optional[ModelService] = None

def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service