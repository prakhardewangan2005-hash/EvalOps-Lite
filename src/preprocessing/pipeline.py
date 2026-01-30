"""
Main preprocessing pipeline
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import logging
from typing import Optional, Tuple

from src.preprocessing.transformers import FeatureScaler, OutlierHandler, FeatureSelector

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for ML models
    """
    
    def __init__(self, config: Optional[dict] = None):
        """
        Initialize preprocessing pipeline
        
        Args:
            config: Configuration dictionary for pipeline
        """
        self.config = config or {
            'imputation_strategy': 'median',
            'scaling_strategy': 'standard',
            'outlier_threshold': 3.0,
            'n_features': 20
        }
        
        self.pipeline: Optional[Pipeline] = None
        self.is_fitted = False
        
    def build_pipeline(self) -> Pipeline:
        """
        Build and return the preprocessing pipeline
        
        Returns:
            sklearn Pipeline object
        """
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=self.config['imputation_strategy'])),
            ('outlier_handler', OutlierHandler(threshold=self.config['outlier_threshold'])),
            ('feature_selector', FeatureSelector(n_features=self.config['n_features'])),
            ('scaler', FeatureScaler(scaling_strategy=self.config['scaling_strategy']))
        ])
        
        logger.info("Built preprocessing pipeline")
        return self.pipeline
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'PreprocessingPipeline':
        """
        Fit the preprocessing pipeline
        
        Args:
            X: Training features
            y: Training labels (unused, for API compatibility)
            
        Returns:
            Self
        """
        if self.pipeline is None:
            self.build_pipeline()
        
        self.pipeline.fit(X)
        self.is_fitted = True
        
        logger.info(f"Fitted preprocessing pipeline on {X.shape[0]} samples")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted pipeline
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transformation")
        
        return self.pipeline.transform(X)
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit and transform features
        
        Args:
            X: Features to transform
            y: Training labels (unused)
            
        Returns:
            Transformed features
        """
        return self.fit(X).transform(X)
    
    def save(self, filepath: str) -> None:
        """
        Save pipeline to disk
        
        Args:
            filepath: Path to save pipeline
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not built yet")
        
        joblib.dump(self.pipeline, filepath)
        logger.info(f"Saved preprocessing pipeline to {filepath}")
    
    def load(self, filepath: str) -> 'PreprocessingPipeline':
        """
        Load pipeline from disk
        
        Args:
            filepath: Path to load pipeline from
            
        Returns:
            Self
        """
        self.pipeline = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded preprocessing pipeline from {filepath}")
        return self
    
    def get_feature_names(self, original_feature_names: list) -> list:
        """
        Get names of selected features
        
        Args:
            original_feature_names: List of all original feature names
            
        Returns:
            List of selected feature names
        """
        if self.pipeline is None or not self.is_fitted:
            raise ValueError("Pipeline must be fitted first")
        
        # Get selected indices from feature selector
        selector = self.pipeline.named_steps['feature_selector']
        selected_indices = selector.selected_indices
        
        # Map to feature names
        return [original_feature_names[i] for i in selected_indices]
