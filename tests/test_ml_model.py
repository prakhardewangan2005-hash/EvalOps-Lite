"""
Tests for ML model module
"""

import pytest
import numpy as np
import joblib
from pathlib import Path
import tempfile

from src.models.ml_model import MLModel
from src.data.schemas import PredictionRequestSchema


def test_ml_model_initialization():
    """Test MLModel initialization with different types"""
    # Test random forest
    rf_model = MLModel(model_type='random_forest')
    assert rf_model.model_type == 'random_forest'
    assert not rf_model.is_trained
    assert rf_model.training_time == 0.0
    
    # Test logistic regression
    lr_model = MLModel(model_type='logistic_regression')
    assert lr_model.model_type == 'logistic_regression'
    
    # Test SVM
    svm_model = MLModel(model_type='svm')
    assert svm_model.model_type == 'svm'
    
    # Test invalid model type
    with pytest.raises(ValueError):
        MLModel(model_type='invalid_type')


def test_ml_model_custom_parameters():
    """Test MLModel with custom parameters"""
    # Test random forest with custom params
    rf_model = MLModel(
        model_type='random_forest',
        n_estimators=50,
        max_depth=5,
        random_state=42
    )
    
    # Check parameters are set
    assert rf_model.model.n_estimators == 50
    assert rf_model.model.max_depth == 5
    assert rf_model.model.random_state == 42


def test_ml_model_training():
    """Test ML model training"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    # Create and train model
    model = MLModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Check model is trained
    assert model.is_trained
    assert model.training_time > 0
    
    # Check metadata
    assert 'model_type' in model.metadata
    assert 'training_samples' in model.metadata
    assert model.metadata['training_samples'] == 100
    assert 'feature_count' in model.metadata
    assert model.metadata['feature_count'] == 10


def test_ml_model_prediction():
    """Test ML model prediction"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 10)
    
    # Train model
    model = MLModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Test prediction
    predictions, probabilities = model.predict(X_test)
    
    # Check predictions shape
    assert predictions.shape == (20,)
    assert probabilities.shape == (20,)
    
    # Check prediction values
    assert set(predictions).issubset({0, 1})
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    
    # Test prediction without probabilities
    predictions_only, _ = model.predict(X_test, return_proba=False)
    assert predictions_only.shape == (20,)


def test_ml_model_single_prediction():
    """Test single prediction with structured output"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 30)
    y_train = np.random.randint(0, 2, 100)
    
    # Train model
    model = MLModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Generate sample features
    sample_features = list(np.random.randn(30))
    
    # Make single prediction
    result = model.predict_single(sample_features)
    
    # Check result structure
    assert 'label' in result
    assert 'probability' in result
    assert 'confidence' in result
    assert 'rationale' in result
    assert 'latency_ms' in result
    assert 'model_type' in result
    
    # Check value ranges
    assert result['label'] in [0, 1]
    assert 0 <= result['probability'] <= 1
    assert 0 <= result['confidence'] <= 1
    assert result['latency_ms'] > 0
    assert isinstance(result['rationale'], str)
    assert len(result['rationale']) > 0


def test_ml_model_evaluation():
    """Test ML model evaluation"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(50, 10)
    y_test = np.random.randint(0, 2, 50)
    
    # Train model
    model = MLModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Check metrics structure
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    
    # Check metric ranges
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    
    # Log loss might be None if no probabilities
    if metrics['log_loss'] is not None:
        assert metrics['log_loss'] >= 0


def test_ml_model_save_load(tmp_path):
    """Test saving and loading ML model"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    
    # Create, train, and save model
    model1 = MLModel(model_type='random_forest', n_estimators=10)
    model1.train(X_train, y_train)
    
    save_path = tmp_path / "model.pkl"
    model1.save(str(save_path))
    
    # Check file exists
    assert save_path.exists()
    
    # Load model
    model2 = MLModel(model_type='random_forest')
    model2.load(str(save_path))
    
    # Check loaded model is trained
    assert model2.is_trained
    assert model2.model_type == 'random_forest'
    
    # Test prediction with loaded model
    X_test = np.random.randn(10, 10)
    pred1, prob1 = model1.predict(X_test)
    pred2, prob2 = model2.predict(X_test)
    
    # Predictions should be identical
    assert np.array_equal(pred1, pred2)
    assert np.allclose(prob1, prob2)


def test_ml_model_with_different_algorithms():
    """Test different ML algorithms"""
    np.random.seed(42)
    
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.randn(20, 10)
    
    algorithms = ['random_forest', 'logistic_regression', 'svm']
    
    for algo in algorithms:
        model = MLModel(model_type=algo)
        model.train(X_train, y_train)
        
        predictions, probabilities = model.predict(X_test)
        
        assert predictions.shape == (20,)
        assert set(predictions).issubset({0, 1})
        
        if probabilities is not None:
            assert probabilities.shape == (20,)
            assert np.all(probabilities >= 0)
            assert np.all(probabilities <= 1)


def test_prediction_request_validation():
    """Test that prediction single validates input"""
    np.random.seed(42)
    
    # Generate synthetic data
    X_train = np.random.randn(100, 30)
    y_train = np.random.randint(0, 2, 100)
    
    # Train model
    model = MLModel(model_type='random_forest', n_estimators=10)
    model.train(X_train, y_train)
    
    # Test with invalid feature length
    invalid_features = list(range(29))  # Should be 30
    
    # Should raise validation error
    with pytest.raises(ValueError):
        model.predict_single(invalid_features)


def test_model_metadata():
    """Test model metadata storage"""
    np.random.seed(42)
    
    X_train = np.random.randn(100, 15)
    y_train = np.random.randint(0, 2, 100)
    
    model = MLModel(model_type='random_forest', n_estimators=20)
    model.train(X_train, y_train)
    
    # Check metadata
    assert model.metadata['model_type'] == 'random_forest'
    assert model.metadata['training_samples'] == 100
    assert model.metadata['feature_count'] == 15
    assert 'training_time_seconds' in model.metadata
    assert model.metadata['training_time_seconds'] > 0
    assert 'classes' in model.metadata
    assert set(model.metadata['classes']).issubset({0, 1})
