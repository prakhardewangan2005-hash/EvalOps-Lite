"""
Application settings and configuration
"""

from pydantic import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "EvalOps Lite"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # Model paths
    model_path: str = "artifacts/model.pkl"
    preprocessor_path: str = "artifacts/preprocessor.pkl"
    evaluation_results_path: str = "artifacts/evaluation_results.json"
    
    # GenAI Model
    genai_model_name: str = "distilbert-base-uncased"
    genai_device: str = "cpu"
    genai_max_length: int = 512
    genai_temperature: float = 0.7
    genai_top_p: float = 0.9
    genai_mock_mode: bool = True  # Use mock for demo
    
    # ML Model
    ml_model_type: str = "random_forest"
    ml_n_estimators: int = 100
    ml_max_depth: int = 10
    
    # Preprocessing
    preprocessing_n_features: int = 20
    preprocessing_scaling_strategy: str = "standard"
    preprocessing_outlier_threshold: float = 3.0
    
    # Evaluation
    evaluation_test_size: float = 0.2
    evaluation_random_state: int = 42
    evaluation_n_latency_samples: int = 100
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    api_workers: int = 1
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
