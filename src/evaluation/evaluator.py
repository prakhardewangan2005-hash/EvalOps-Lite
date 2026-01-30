"""
Model evaluator for comprehensive performance assessment
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import time
import logging
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss
)

from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float = None
    log_loss: float = None
    inference_latency_p50: float = None
    inference_latency_p95: float = None
    inference_latency_p99: float = None
    error_rate: float = None


class ModelEvaluator:
    """
    Evaluator for comprehensive model assessment
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_history: List[Dict[str, Any]] = []
        
    def evaluate_ml_model(
        self, 
        model: MLModel, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        calculate_latency: bool = True,
        n_latency_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate ML model
        
        Args:
            model: ML model to evaluate
            X_test: Test features
            y_test: Test labels
            calculate_latency: Whether to calculate inference latency
            n_latency_samples: Number of samples for latency calculation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating ML model on {X_test.shape[0]} samples")
        
        # Get predictions
        y_pred, y_proba = model.predict(X_test)
        
        # Calculate basic metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Calculate latency if requested
        if calculate_latency:
            latency_metrics = self._calculate_inference_latency(
                model, X_test[:n_latency_samples]
            )
            metrics.update(latency_metrics)
        
        # Prepare results
        results = {
            'model_type': 'ml_model',
            'model_name': model.model_type,
            'metrics': metrics,
            'test_samples': X_test.shape[0],
            'timestamp': time.time()
        }
        
        self.metrics_history.append(results)
        
        return results
    
    def evaluate_genai_model(
        self,
        model: GenAIBaseline,
        X_test: np.ndarray,
        y_test: np.ndarray,
        calculate_latency: bool = True,
        n_latency_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate GenAI model
        
        Args:
            model: GenAI model to evaluate
            X_test: Test features
            y_test: Test labels
            calculate_latency: Whether to calculate inference latency
            n_latency_samples: Number of samples for latency calculation
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating GenAI model on {X_test.shape[0]} samples")
        
        # Convert numpy array to list of lists for GenAI model
        X_test_list = X_test.tolist()
        
        # Get predictions
        predictions = []
        latencies = []
        
        for features in X_test_list:
            start_time = time.time()
            pred = model.predict(features)
            latency = (time.time() - start_time) * 1000
            
            predictions.append(pred)
            latencies.append(latency)
        
        # Extract labels and probabilities
        y_pred = np.array([p['label'] for p in predictions])
        y_proba = np.array([p['probability'] for p in predictions])
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_proba)
        
        # Add latency metrics
        if calculate_latency:
            latency_metrics = self._calculate_latency_from_list(latencies)
            metrics.update(latency_metrics)
        
        # Check for schema violations
        schema_violations = self._check_schema_violations(predictions)
        metrics['schema_violations'] = schema_violations
        
        # Prepare results
        results = {
            'model_type': 'genai_model',
            'model_name': model.config.model_name,
            'metrics': metrics,
            'test_samples': X_test.shape[0],
            'schema_violations': schema_violations,
            'timestamp': time.time()
        }
        
        self.metrics_history.append(results)
        
        return results
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'error_rate': 1 - accuracy_score(y_true, y_pred)
        }
        
        # Add probability-based metrics if available
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception as e:
                logger.warning(f"Could not calculate probability metrics: {str(e)}")
                metrics['roc_auc'] = 0.5
                metrics['log_loss'] = None
        
        return metrics
    
    def _calculate_inference_latency(
        self, 
        model: MLModel, 
        X_samples: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate inference latency metrics
        
        Args:
            model: Model to test
            X_samples: Samples for latency testing
            
        Returns:
            Dictionary with latency metrics
        """
        latencies = []
        
        for i in range(X_samples.shape[0]):
            sample = X_samples[i:i+1]
            start_time = time.time()
            model.predict(sample, return_proba=True)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        return self._calculate_latency_from_list(latencies)
    
    def _calculate_latency_from_list(self, latencies: List[float]) -> Dict[str, float]:
        """
        Calculate latency percentiles from list
        
        Args:
            latencies: List of latency measurements
            
        Returns:
            Dictionary with latency percentiles
        """
        latencies_array = np.array(latencies)
        
        return {
            'inference_latency_p50': float(np.percentile(latencies_array, 50)),
            'inference_latency_p95': float(np.percentile(latencies_array, 95)),
            'inference_latency_p99': float(np.percentile(latencies_array, 99)),
            'inference_latency_mean': float(np.mean(latencies_array)),
            'inference_latency_std': float(np.std(latencies_array)),
            'inference_latency_min': float(np.min(latencies_array)),
            'inference_latency_max': float(np.max(latencies_array)),
            'n_latency_samples': len(latencies)
        }
    
    def _check_schema_violations(self, predictions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Check for schema violations in GenAI outputs
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Dictionary with violation counts
        """
        violations = {
            'label_range': 0,
            'probability_range': 0,
            'confidence_range': 0,
            'rationale_length': 0,
            'missing_fields': 0,
            'total_predictions': len(predictions)
        }
        
        for pred in predictions:
            # Check label range
            if not (0 <= pred.get('label', -1) <= 1):
                violations['label_range'] += 1
            
            # Check probability range
            prob = pred.get('probability', -1)
            if not (0 <= prob <= 1):
                violations['probability_range'] += 1
            
            # Check confidence range
            conf = pred.get('confidence', -1)
            if not (0 <= conf <= 1):
                violations['confidence_range'] += 1
            
            # Check rationale length
            rationale = pred.get('rationale', '')
            if len(rationale) < 10:
                violations['rationale_length'] += 1
            
            # Check missing required fields
            required_fields = ['label', 'probability', 'confidence', 'rationale', 'latency_ms']
            missing = [field for field in required_fields if field not in pred]
            if missing:
                violations['missing_fields'] += 1
        
        return violations
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations"""
        return self.metrics_history
    
    def clear_history(self):
        """Clear evaluation history"""
        self.metrics_history.clear()
