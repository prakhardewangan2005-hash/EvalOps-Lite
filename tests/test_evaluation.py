"""
Tests for evaluation module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.metrics import MetricsTracker
from src.evaluation.comparator import ModelComparator
from src.evaluation.safety import SafetyValidator


def test_model_evaluator_initialization():
    """Test ModelEvaluator initialization"""
    evaluator = ModelEvaluator()
    
    assert evaluator.metrics_history == []
    assert len(evaluator.metrics_history) == 0


def test_calculate_metrics():
    """Test metrics calculation"""
    evaluator = ModelEvaluator()
    
    # Create sample data
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])  # One wrong prediction
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.6, 0.8, 0.3, 0.7])
    
    metrics = evaluator._calculate_metrics(y_true, y_pred, y_proba)
    
    # Check all metrics present
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'error_rate' in metrics
    assert 'roc_auc' in metrics
    assert 'log_loss' in metrics
    
    # Check metric values
    # With one error in 8 samples, accuracy should be 0.875
    assert metrics['accuracy'] == 0.875
    assert metrics['error_rate'] == 0.125
    
    # Check ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1
    assert metrics['log_loss'] >= 0


def test_calculate_metrics_without_probabilities():
    """Test metrics calculation without probabilities"""
    evaluator = ModelEvaluator()
    
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    metrics = evaluator._calculate_metrics(y_true, y_pred, None)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # ROC AUC and log loss should not be present or should be default
    if 'roc_auc' in metrics:
        assert metrics['roc_auc'] == 0.5
    if 'log_loss' in metrics:
        assert metrics['log_loss'] is None


def test_calculate_latency_from_list():
    """Test latency calculation from list"""
    evaluator = ModelEvaluator()
    
    # Create sample latencies
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    
    latency_metrics = evaluator._calculate_latency_from_list(latencies)
    
    # Check all metrics present
    assert 'inference_latency_p50' in latency_metrics
    assert 'inference_latency_p95' in latency_metrics
    assert 'inference_latency_p99' in latency_metrics
    assert 'inference_latency_mean' in latency_metrics
    assert 'inference_latency_std' in latency_metrics
    assert 'inference_latency_min' in latency_metrics
    assert 'inference_latency_max' in latency_metrics
    assert 'n_latency_samples' in latency_metrics
    
    # Check values
    assert latency_metrics['inference_latency_p50'] == 30.0
    assert latency_metrics['inference_latency_mean'] == 30.0
    assert latency_metrics['inference_latency_min'] == 10.0
    assert latency_metrics['inference_latency_max'] == 50.0
    assert latency_metrics['n_latency_samples'] == 5


def test_check_schema_violations():
    """Test schema violation checking"""
    evaluator = ModelEvaluator()
    
    # Create sample predictions
    predictions = [
        {
            'label': 0,
            'probability': 0.3,
            'confidence': 0.4,
            'rationale': 'Valid rationale with sufficient length',
            'latency_ms': 10.0
        },
        {
            'label': 2,  # Invalid: should be 0 or 1
            'probability': 1.5,  # Invalid: > 1
            'confidence': -0.1,  # Invalid: < 0
            'rationale': 'short',  # Invalid: too short
            'latency_ms': 10.0
        },
        {
            # Missing fields
            'label': 1,
            'probability': 0.8
        }
    ]
    
    violations = evaluator._check_schema_violations(predictions)
    
    assert violations['total_predictions'] == 3
    assert violations['label_range'] == 1  # Second prediction has label 2
    assert violations['probability_range'] == 1  # Second prediction has probability 1.5
    assert violations['confidence_range'] == 1  # Second prediction has confidence -0.1
    assert violations['rationale_length'] == 1  # Second prediction has short rationale
    assert violations['missing_fields'] == 1  # Third prediction missing fields


def test_evaluate_ml_model():
    """Test ML model evaluation"""
    evaluator = ModelEvaluator()
    
    # Mock ML model
    mock_model = Mock()
    mock_model.model_type = 'random_forest'
    mock_model.is_trained = True
    
    # Mock prediction method
    predictions = np.array([0, 1, 0, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
    mock_model.predict.return_value = (predictions, probabilities)
    
    # Create test data
    X_test = np.random.randn(5, 10)
    y_test = np.array([0, 1, 0, 1, 0])
    
    # Mock latency calculation
    with patch.object(evaluator, '_calculate_inference_latency') as mock_latency:
        mock_latency.return_value = {
            'inference_latency_p50': 10.0,
            'inference_latency_p95': 20.0,
            'inference_latency_p99': 30.0
        }
        
        results = evaluator.evaluate_ml_model(
            mock_model, X_test, y_test,
            calculate_latency=True,
            n_latency_samples=3
        )
    
    # Check results structure
    assert results['model_type'] == 'ml_model'
    assert results['model_name'] == 'random_forest'
    assert 'metrics' in results
    assert results['test_samples'] == 5
    assert 'timestamp' in results
    
    # Check metrics
    metrics = results['metrics']
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check latency metrics
    assert 'inference_latency_p50' in metrics
    assert 'inference_latency_p95' in metrics
    assert 'inference_latency_p99' in metrics


def test_metrics_tracker_initialization(tmp_path):
    """Test MetricsTracker initialization"""
    tracker = MetricsTracker(output_dir=str(tmp_path))
    
    assert tracker.output_dir == tmp_path
    assert tracker.metrics_history == []


def test_metrics_tracker_add_metrics():
    """Test adding metrics to tracker"""
    tracker = MetricsTracker(output_dir=".")
    
    # Add metrics
    metrics = {'accuracy': 0.95, 'f1_score': 0.93}
    tracker.add_metrics(metrics, 'test_model', 'evaluation')
    
    assert len(tracker.metrics_history) == 1
    
    entry = tracker.metrics_history[0]
    assert entry['model_name'] == 'test_model'
    assert entry['eval_type'] == 'evaluation'
    assert entry['metrics']['accuracy'] == 0.95
    assert entry['metrics']['f1_score'] == 0.93
    assert 'timestamp' in entry


def test_metrics_tracker_save_load(tmp_path):
    """Test saving and loading metrics"""
    tracker = MetricsTracker(output_dir=str(tmp_path))
    
    # Add some metrics
    tracker.add_metrics({'accuracy': 0.9}, 'model1', 'test1')
    tracker.add_metrics({'accuracy': 0.95}, 'model2', 'test2')
    
    # Save metrics
    tracker.save_metrics('test_metrics.json')
    
    # Create new tracker and load
    tracker2 = MetricsTracker(output_dir=str(tmp_path))
    tracker2.load_metrics('test_metrics.json')
    
    assert len(tracker2.metrics_history) == 2
    assert tracker2.metrics_history[0]['model_name'] == 'model1'
    assert tracker2.metrics_history[1]['model_name'] == 'model2'


def test_metrics_tracker_get_latest_metrics():
    """Test getting latest metrics"""
    tracker = MetricsTracker(output_dir=".")
    
    # Add metrics
    tracker.add_metrics({'accuracy': 0.9}, 'model1', 'test')
    tracker.add_metrics({'accuracy': 0.95}, 'model1', 'test')
    tracker.add_metrics({'accuracy': 0.8}, 'model2', 'test')
    
    # Get latest for model1
    latest = tracker.get_latest_metrics('model1')
    assert latest['metrics']['accuracy'] == 0.95
    
    # Get latest overall
    latest_overall = tracker.get_latest_metrics()
    assert latest_overall['model_name'] == 'model2'
    assert latest_overall['metrics']['accuracy'] == 0.8


def test_safety_validator_initialization():
    """Test SafetyValidator initialization"""
    validator = SafetyValidator()
    
    assert validator.violation_history == []


def test_safety_validator_validate_genai_output():
    """Test GenAI output safety validation"""
    validator = SafetyValidator()
    
    # Valid output
    valid_output = {
        'label': 1,
        'probability': 0.85,
        'confidence': 0.7,
        'rationale': 'This is a sufficiently long rationale that explains the prediction.',
        'latency_ms': 45.2
    }
    
    result = validator.validate_genai_output(valid_output)
    
    assert result.passed == True
    assert len(result.violations) == 0
    assert result.score == 1.0
    
    # Invalid output
    invalid_output = {
        'label': 2,  # Invalid
        'probability': 1.5,  # Invalid
        'confidence': 1.2,  # Invalid
        'rationale': 'short',  # Too short
        'latency_ms': 0.1  # Too fast
    }
    
    result = validator.validate_genai_output(invalid_output)
    
    assert result.passed == False
    assert len(result.violations) > 0
    assert result.score < 1.0


def test_model_comparator_initialization(tmp_path):
    """Test ModelComparator initialization"""
    comparator = ModelComparator(output_dir=str(tmp_path))
    
    assert isinstance(comparator.evaluator, ModelEvaluator)
    assert isinstance(comparator.safety_validator, SafetyValidator)
    assert isinstance(comparator.metrics_tracker, MetricsTracker)
    assert comparator.comparison_history == []


def test_perform_comparison():
    """Test comparison between model results"""
    comparator = ModelComparator()
    
    # Create mock results
    ml_results = {
        'model_type': 'ml_model',
        'model_name': 'random_forest',
        'metrics': {
            'accuracy': 0.95,
            'f1_score': 0.94,
            'inference_latency_p95': 10.0
        }
    }
    
    genai_results = {
        'model_type': 'genai_model',
        'model_name': 'distilbert',
        'metrics': {
            'accuracy': 0.92,
            'f1_score': 0.91,
            'inference_latency_p95': 50.0
        }
    }
    
    comparison = comparator._perform_comparison(ml_results, genai_results)
    
    assert 'accuracy_difference' in comparison
    assert 'f1_difference' in comparison
    assert 'latency_ratio' in comparison
    assert 'performance_score' in comparison
    assert 'key_metrics' in comparison
    
    # Check calculations
    assert comparison['accuracy_difference'] == -0.03  # genai - ml
    assert comparison['latency_ratio'] == 5.0  # 50.0 / 10.0
    
    # Check performance scores
    perf_scores = comparison['performance_score']
    assert 'ml_performance_score' in perf_scores
    assert 'genai_performance_score' in perf_scores
    assert 'performance_ratio' in perf_scores


def test_determine_winner():
    """Test determining comparison winner"""
    comparator = ModelComparator()
    
    # ML clearly better
    comparison1 = {
        'performance_score': {
            'ml_performance_score': 0.9,
            'genai_performance_score': 0.7,
            'performance_ratio': 0.78
        }
    }
    
    winner1 = comparator._determine_winner(comparison1)
    assert winner1 == "ML_MODEL"
    
    # GenAI clearly better
    comparison2 = {
        'performance_score': {
            'ml_performance_score': 0.7,
            'genai_performance_score': 0.9,
            'performance_ratio': 1.29
        }
    }
    
    winner2 = comparator._determine_winner(comparison2)
    assert winner2 == "GENAI_MODEL"
    
    # Tie (within 10%)
    comparison3 = {
        'performance_score': {
            'ml_performance_score': 0.8,
            'genai_performance_score': 0.85,
            'performance_ratio': 1.06
        }
    }
    
    winner3 = comparator._determine_winner(comparison3)
    assert winner3 == "TIE"


def test_determine_winner_with_safety():
    """Test determining winner with safety considerations"""
    comparator = ModelComparator()
    
    comparison = {
        'performance_score': {
            'ml_performance_score': 0.8,
            'genai_performance_score': 0.9,
            'performance_ratio': 1.125
        }
    }
    
    # GenAI would win without safety penalty
    winner_no_safety = comparator._determine_winner(comparison)
    assert winner_no_safety == "GENAI_MODEL"
    
    # With high violation rate, GenAI gets penalty
    safety_report = {
        'violation_rate': 0.2,  # 20% violation rate
        'average_score': 0.8
    }
    
    winner_with_safety = comparator._determine_winner(comparison, safety_report)
    # With 20% penalty, GenAI score becomes 0.9 * 0.8 = 0.72, ML wins
    assert winner_with_safety == "ML_MODEL"
