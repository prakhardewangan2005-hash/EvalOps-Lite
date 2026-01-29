"""
Custom transformers for preprocessing pipeline
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """
    Custom feature scaler with optional scaling strategy
    """
    
    def __init__(self, scaling_strategy: str = 'standard'):
        """
        Initialize feature scaler
        
        Args:
            scaling_strategy: 'standard' for StandardScaler, 
                            'robust' for RobustScaler
        """
        self.scaling_strategy = scaling_strategy
        if scaling_strategy == 'standard':
            self.scaler = StandardScaler()
        elif scaling_strategy == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling strategy: {scaling_strategy}")
        
    def fit(self, X: np.ndarray, y=None):
        """Fit the scaler"""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features"""
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit and transform features"""
        return self.scaler.fit_transform(X)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handle outliers using IQR method
    """
    
    def __init__(self, threshold: float = 3.0):
        """
        Initialize outlier handler
        
        Args:
            threshold: IQR multiplier for outlier detection
        """
        self.threshold = threshold
        self.lower_bounds = None
        self.upper_bounds = None
        
    def fit(self, X: np.ndarray, y=None):
        """Calculate IQR bounds"""
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        self.lower_bounds = Q1 - self.threshold * IQR
        self.upper_bounds = Q3 + self.threshold * IQR
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Clip outliers to bounds"""
        X_transformed = X.copy()
        
        for i in range(X.shape[1]):
            mask_lower = X[:, i] < self.lower_bounds[i]
            mask_upper = X[:, i] > self.upper_bounds[i]
            
            X_transformed[mask_lower, i] = self.lower_bounds[i]
            X_transformed[mask_upper, i] = self.upper_bounds[i]
            
            if np.any(mask_lower) or np.any(mask_upper):
                logger.debug(f"Feature {i}: Clipped {np.sum(mask_lower) + np.sum(mask_upper)} outliers")
        
        return X_transformed


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select top-k features based on variance
    """
    
    def __init__(self, n_features: int = 20):
        """
        Initialize feature selector
        
        Args:
            n_features: Number of top features to select
        """
        self.n_features = n_features
        self.selected_indices = None
        
    def fit(self, X: np.ndarray, y=None):
        """Select features based on variance"""
        variances = np.var(X, axis=0)
        self.selected_indices = np.argsort(variances)[-self.n_features:]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Select features"""
        return X[:, self.selected_indices]
