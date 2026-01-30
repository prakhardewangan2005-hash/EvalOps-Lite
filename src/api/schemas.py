"""
API request/response schemas
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    label: int = Field(..., ge=0, le=1, description="Predicted class")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence")
    rationale: str = Field(..., description="Explanation for prediction")
    latency_ms: float = Field(..., gt=0, description="Inference latency in milliseconds")
    model_type: Optional[str] = Field(None, description="Type of model used")
    
    class Config:
        schema_extra = {
            "example": {
                "label": 1,
                "probability": 0.85,
                "confidence": 0.7,
                "rationale": "High probability based on feature analysis",
                "latency_ms": 45.2,
                "model_type": "random_forest"
            }
        }


class CombinedPredictionResponse(BaseModel):
    """Schema for combined ML and GenAI predictions"""
    request_id: str = Field(..., description="Unique request identifier")
    features: List[float] = Field(..., description="Input features")
    ml_prediction: PredictionResponse = Field(..., description="ML model prediction")
    genai_prediction: PredictionResponse = Field(..., description="GenAI model prediction")
    agreement: bool = Field(..., description="Whether predictions agree")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "request_id": "req_123456",
                "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189],
                "ml_prediction": {
                    "label": 1,
                    "probability": 0.92,
                    "confidence": 0.84,
                    "rationale": "Random forest prediction",
                    "latency_ms": 12.5,
                    "model_type": "random_forest"
                },
                "genai_prediction": {
                    "label": 1,
                    "probability": 0.88,
                    "confidence": 0.76,
                    "rationale": "Transformer-based analysis",
                    "latency_ms": 245.3,
                    "model_type": "distilbert"
                },
                "agreement": True,
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests"""
    features_list: List[List[float]] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of feature lists for batch prediction"
    )
    
    @validator('features_list')
    def validate_features_length(cls, v):
        for features in v:
            if len(features) != 30:
                raise ValueError(f"Each feature list must have 30 values, got {len(features)}")
        return v


class MetricsResponse(BaseModel):
    """Schema for metrics response"""
    status: str = Field(..., description="Response status")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of latest metrics")
    latest_evaluations: Optional[List[Dict[str, Any]]] = Field(None, description="Latest evaluations")
    latency_metrics: Optional[Dict[str, Any]] = Field(None, description="Latency metrics")
    total_evaluations: Optional[int] = Field(None, description="Total number of evaluations")
    model_status: Optional[Dict[str, Any]] = Field(None, description="Model registry status")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "timestamp": "2024-01-01T12:00:00Z",
                "latest_evaluations": [
                    {
                        "timestamp": "2024-01-01T12:00:00Z",
                        "model_name": "ml_random_forest",
                        "eval_type": "comparison",
                        "metrics": {
                            "accuracy": 0.95,
                            "precision": 0.96,
                            "recall": 0.94,
                            "f1_score": 0.95,
                            "inference_latency_p95": 15.2
                        }
                    }
                ],
                "latency_metrics": {
                    "ml_random_forest_comparison": {
                        "p95_latency_ms": 15.2,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                },
                "total_evaluations": 42,
                "model_status": {
                    "ml_model": {"loaded": True},
                    "genai_model": {"loaded": True}
                }
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    model_registry: Optional[Dict[str, Any]] = Field(None, description="Model registry status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "model_registry": {
                    "ml_model": {"loaded": True},
                    "genai_model": {"loaded": True}
                },
                "service": "EvalOps Lite API",
                "version": "1.0.0"
            }
        }
