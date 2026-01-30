"""
Dependency injection for FastAPI routes
"""

from fastapi import Depends, HTTPException
from typing import Dict, Any

from src.models.registry import ModelRegistry
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline
from src.api.app import model_registry


def get_model_registry() -> ModelRegistry:
    """
    Get model registry dependency
    
    Returns:
        ModelRegistry instance
        
    Raises:
        HTTPException: If registry not initialized
    """
    if model_registry is None:
        raise HTTPException(
            status_code=503,
            detail="Model registry not initialized"
        )
    
    return model_registry


def get_ml_model(registry: ModelRegistry = Depends(get_model_registry)) -> MLModel:
    """
    Get ML model dependency
    
    Args:
        registry: Model registry
        
    Returns:
        MLModel instance
        
    Raises:
        HTTPException: If ML model not loaded
    """
    if registry.ml_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded"
        )
    
    return registry.ml_model


def get_genai_model(registry: ModelRegistry = Depends(get_model_registry)) -> GenAIBaseline:
    """
    Get GenAI model dependency
    
    Args:
        registry: Model registry
        
    Returns:
        GenAIBaseline instance
        
    Raises:
        HTTPException: If GenAI model not loaded
    """
    if registry.genai_model is None:
        raise HTTPException(
            status_code=503,
            detail="GenAI model not loaded"
        )
    
    return registry.genai_model


def get_preprocessor(registry: ModelRegistry = Depends(get_model_registry)) -> Any:
    """
    Get preprocessor dependency
    
    Args:
        registry: Model registry
        
    Returns:
        Preprocessor instance
        
    Raises:
        HTTPException: If preprocessor not loaded
    """
    if registry.preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessor not loaded"
        )
    
    return registry.preprocessor
