"""
Data ingestion module for loading and validating datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from src.data.schemas import DataSchema

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """Container for dataset splits"""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list
    target_names: list
    metadata: Dict[str, Any]


class DataIngestor:
    """Handles data loading, validation, and splitting"""
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize data ingestor
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.schema = DataSchema()
        self.dataset: Optional[Dataset] = None
        
    def load_breast_cancer_data(self) -> Dataset:
        """
        Load and prepare breast cancer dataset for binary classification
        
        Returns:
            Dataset object with train/test splits
        """
        try:
            # Load dataset from sklearn
            data = load_breast_cancer()
            X = data.data
            y = data.target
            feature_names = data.feature_names.tolist()
            target_names = data.target_names.tolist()
            
            # Convert to pandas for validation
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Validate schema
            self.schema.validate(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y
            )
            
            # Create dataset object
            self.dataset = Dataset(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_names,
                target_names=target_names,
                metadata={
                    'dataset_name': 'breast_cancer',
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'n_classes': len(np.unique(y)),
                    'class_distribution': np.bincount(y).tolist(),
                    'test_size': self.test_size,
                    'random_state': self.random_state
                }
            )
            
            logger.info(f"Loaded breast cancer dataset with {len(X)} samples")
            logger.info(f"Training set: {X_train.shape[0]} samples")
            logger.info(f"Test set: {X_test.shape[0]} samples")
            logger.info(f"Feature names: {feature_names}")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics for the dataset"""
        if self.dataset is None:
            raise ValueError("No dataset loaded. Call load_breast_cancer_data() first.")
        
        stats = {
            'training_samples': self.dataset.X_train.shape[0],
            'test_samples': self.dataset.X_test.shape[0],
            'n_features': self.dataset.X_train.shape[1],
            'n_classes': len(np.unique(self.dataset.y_train)),
            'train_class_distribution': np.bincount(self.dataset.y_train).tolist(),
            'test_class_distribution': np.bincount(self.dataset.y_test).tolist(),
            'feature_names': self.dataset.feature_names,
            'target_names': self.dataset.target_names
        }
        
        return stats
