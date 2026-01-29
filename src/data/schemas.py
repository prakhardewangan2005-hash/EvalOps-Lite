"""
Data schema definitions and validation
"""

from typing import Optional, List
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np


class DataSchema(BaseModel):
    """Schema for breast cancer dataset validation"""
    
    class Config:
        arbitrary_types_allowed = True
    
    def validate(self, df: pd.DataFrame) -> bool:
        """
        Validate dataframe against schema
        
        Args:
            df: Input dataframe
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns (features + target)
        required_columns = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area',
            'mean smoothness', 'mean compactness', 'mean concavity',
            'mean concave points', 'mean symmetry', 'mean fractal dimension',
            'radius error', 'texture error', 'perimeter error', 'area error',
            'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension',
            'target'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check data types
        for col in required_columns[:-1]:  # All features should be numeric
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column {col} must be numeric")
        
        # Check target values (0 or 1)
        if not df['target'].isin([0, 1]).all():
            raise ValueError("Target column must contain only 0 or 1")
        
        # Check for NaN values
        if df[required_columns].isnull().any().any():
            raise ValueError("Dataset contains NaN values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if np.any(np.isinf(df[numeric_cols].values)):
            raise ValueError("Dataset contains infinite values")
        
        return True


class PredictionRequestSchema(BaseModel):
    """Schema for prediction requests"""
    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="30 feature values for breast cancer prediction"
    )
    
    @validator('features')
    def validate_features_length(cls, v):
        if len(v) != 30:
            raise ValueError(f"Expected 30 features, got {len(v)}")
        return v


class GenAIOutputSchema(BaseModel):
    """Schema for GenAI model outputs"""
    label: int = Field(..., ge=0, le=1, description="Predicted class (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    rationale: str = Field(..., min_length=10, description="Explanation for prediction")
    latency_ms: float = Field(..., gt=0, description="Inference latency in milliseconds")
    
    @validator('probability', 'confidence')
    def validate_probability_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError(f"Value must be between 0 and 1, got {v}")
        return v
