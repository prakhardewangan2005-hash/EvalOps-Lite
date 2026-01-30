"""
Classic ML model implementation using scikit-learn
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import logging
from typing import Optional, Dict, Any, Tuple
import time

from src.data.schemas import PredictionRequestSchema

logger = logging.getLogger(__name__)


class MLModel:
    """
    Classic ML model with configurable algorithms
    """
    
    def __init__(self, model_type: str = 'random_forest', **kwargs):
        """
        Initialize ML model
        
        Args:
            model_type: Type of model ('random_forest', 'logistic_regression', 'svm')
            **kwargs: Additional model parameters
        """
        self.model_type = model_type
        self.model = self._build_model(model_type, kwargs)
        self.is_trained = False
        self.training_time = 0.0
        self.metadata: Dict[str, Any] = {}
        
    def _build_model(self, model_type: str, params: dict):
        """Build model based on type"""
        if model_type == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'n_jobs': -1
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
        elif model_type == 'logistic_regression':
            default_params = {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42,
                'solver': 'lbfgs'
            }
            default_params.update(params)
            return LogisticRegression(**default_params)
        
        elif model_type == 'svm':
            default_params = {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            }
            default_params.update(params)
            # Calibrate SVM for better probability estimates
            base_svm = SVC(**default_params)
            return CalibratedClassifierCV(base_svm, cv=3)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> 'MLModel':
        """
        Train the ML model
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Self
        """
        start_time = time.time()
        
        logger.info(f"Training {self.model_type} on {X_train.shape[0]} samples...")
        
        try:
            self.model.fit(X_train, y_train)
            self.is_trained = True
            self.training_time = time.time() - start_time
            
            # Store metadata
            self.metadata.update({
                'model_type': self.model_type,
                'training_samples': X_train.shape[0],
                'training_time_seconds': self.training_time,
                'feature_count': X_train.shape[1],
                'classes': np.unique(y_train).tolist()
            })
            
            logger.info(f"Model trained in {self.training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray, return_proba: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions
        
        Args:
            X: Input features
            return_proba: Whether to return probabilities
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        start_time = time.time()
        
        try:
            predictions = self.model.predict(X)
            probabilities = None
            
            if return_proba:
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X)[:, 1]
                else:
                    # For models without predict_proba, use decision function
                    if hasattr(self.model, 'decision_function'):
                        decision_scores = self.model.decision_function(X)
                        # Convert to probabilities using sigmoid
                        probabilities = 1 / (1 + np.exp(-decision_scores))
                    else:
                        probabilities = predictions.astype(float)
            
            latency_ms = (time.time() - start_time) * 1000
            logger.debug(f"Prediction completed in {latency_ms:.2f}ms")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_single(self, features: list) -> Dict[str, Any]:
        """
        Predict for a single sample with structured output
        
        Args:
            features: List of feature values
            
        Returns:
            Dictionary with prediction results
        """
        # Validate input
        schema = PredictionRequestSchema(features=features)
        
        # Convert to numpy array
        X = np.array(schema.features).reshape(1, -1)
        
        # Make prediction
        start_time = time.time()
        predictions, probabilities = self.predict(X)
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare response
        result = {
            'label': int(predictions[0]),
            'probability': float(probabilities[0]) if probabilities is not None else 0.5,
            'confidence': float(probabilities[0]) if probabilities is not None else 0.5,
            'rationale': f"Classic ML ({self.model_type}) prediction based on {X.shape[1]} features",
            'latency_ms': latency_ms,
            'model_type': self.model_type
        }
        
        return result
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        predictions, probabilities = self.predict(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions, average='binary'),
            'recall': recall_score(y_test, predictions, average='binary'),
            'f1_score': f1_score(y_test, predictions, average='binary'),
            'log_loss': None  # Will be calculated if probabilities available
        }
        
        if probabilities is not None:
            from sklearn.metrics import log_loss
            metrics['log_loss'] = log_loss(y_test, probabilities)
        
        logger.info(f"Model evaluation - Accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        joblib.dump({
            'model': self.model,
            'metadata': self.metadata,
            'model_type': self.model_type
        }, filepath)
        
        logger.info(f"Saved model to {filepath}")
    
    def load(self, filepath: str) -> 'MLModel':
        """
        Load model from disk
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.metadata = data['metadata']
        self.model_type = data['model_type']
        self.is_trained = True
        
        logger.info(f"Loaded model from {filepath}")
        return self
