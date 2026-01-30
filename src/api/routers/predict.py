"""
Prediction router for ML and GenAI models
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import numpy as np
import logging

from src.api.dependencies import get_ml_model, get_genai_model, get_preprocessor
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline
from src.data.schemas import PredictionRequestSchema
from src.api.schemas import (
    PredictionResponse,
    CombinedPredictionResponse,
    BatchPredictionRequest
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=CombinedPredictionResponse)
async def predict(
    request: PredictionRequestSchema,
    ml_model: MLModel = Depends(get_ml_model),
    genai_model: GenAIBaseline = Depends(get_genai_model),
    preprocessor = Depends(get_preprocessor)
) -> Dict[str, Any]:
    """
    Get predictions from both ML and GenAI models
    
    Args:
        request: Prediction request with features
        ml_model: ML model instance
        genai_model: GenAI model instance
        preprocessor: Preprocessor instance
        
    Returns:
        Combined predictions from both models
    """
    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # Preprocess features for ML model
        features_processed = preprocessor.transform(features_array)
        
        # Get ML prediction
        ml_prediction = ml_model.predict_single(features_processed[0].tolist())
        
        # Get GenAI prediction (uses raw features)
        genai_prediction = genai_model.predict(request.features)
        
        # Prepare response
        response = CombinedPredictionResponse(
            request_id=f"req_{hash(tuple(request.features)) % 1000000:06d}",
            features=request.features,
            ml_prediction=PredictionResponse(**ml_prediction),
            genai_prediction=PredictionResponse(**genai_prediction),
            agreement=ml_prediction['label'] == genai_prediction['label']
        )
        
        logger.info(f"Prediction completed - ML: {ml_prediction['label']}, "
                   f"GenAI: {genai_prediction['label']}, "
                   f"Agreement: {response.agreement}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/predict/batch", response_model=List[CombinedPredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    ml_model: MLModel = Depends(get_ml_model),
    genai_model: GenAIBaseline = Depends(get_genai_model),
    preprocessor = Depends(get_preprocessor)
) -> List[Dict[str, Any]]:
    """
    Get batch predictions from both models
    
    Args:
        request: Batch prediction request
        ml_model: ML model instance
        genai_model: GenAI model instance
        preprocessor: Preprocessor instance
        
    Returns:
        List of combined predictions
    """
    try:
        responses = []
        
        for i, features in enumerate(request.features_list):
            # Validate each feature set
            feature_request = PredictionRequestSchema(features=features)
            
            # Convert to numpy array
            features_array = np.array(feature_request.features).reshape(1, -1)
            
            # Preprocess for ML model
            features_processed = preprocessor.transform(features_array)
            
            # Get ML prediction
            ml_prediction = ml_model.predict_single(features_processed[0].tolist())
            
            # Get GenAI prediction
            genai_prediction = genai_model.predict(feature_request.features)
            
            # Create response
            response = CombinedPredictionResponse(
                request_id=f"batch_{i:04d}_{hash(tuple(features)) % 10000:04d}",
                features=features,
                ml_prediction=PredictionResponse(**ml_prediction),
                genai_prediction=PredictionResponse(**genai_prediction),
                agreement=ml_prediction['label'] == genai_prediction['label']
            )
            
            responses.append(response)
        
        logger.info(f"Batch prediction completed - {len(responses)} predictions")
        
        return responses
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


@router.get("/predict/ml", response_model=PredictionResponse)
async def predict_ml_only(
    request: PredictionRequestSchema,
    ml_model: MLModel = Depends(get_ml_model),
    preprocessor = Depends(get_preprocessor)
) -> Dict[str, Any]:
    """
    Get prediction only from ML model
    
    Args:
        request: Prediction request
        ml_model: ML model instance
        preprocessor: Preprocessor instance
        
    Returns:
        ML model prediction
    """
    try:
        features_array = np.array(request.features).reshape(1, -1)
        features_processed = preprocessor.transform(features_array)
        
        prediction = ml_model.predict_single(features_processed[0].tolist())
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"ML prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"ML prediction failed: {str(e)}"
        )


@router.get("/predict/genai", response_model=PredictionResponse)
async def predict_genai_only(
    request: PredictionRequestSchema,
    genai_model: GenAIBaseline = Depends(get_genai_model)
) -> Dict[str, Any]:
    """
    Get prediction only from GenAI model
    
    Args:
        request: Prediction request
        genai_model: GenAI model instance
        
    Returns:
        GenAI model prediction
    """
    try:
        prediction = genai_model.predict(request.features)
        
        return PredictionResponse(**prediction)
        
    except Exception as e:
        logger.error(f"GenAI prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"GenAI prediction failed: {str(e)}"
        )
