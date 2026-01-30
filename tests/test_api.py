"""
Tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import json
from unittest.mock import Mock, patch

from src.api.app import create_app, model_registry
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline, GenAIConfig
from src.preprocessing.pipeline import PreprocessingPipeline


@pytest.fixture
def client():
    """Create test client"""
    # Create app
    app = create_app()
    
    # Mock model registry in app state
    with patch('src.api.app.model_registry') as mock_registry:
        # Create mock models
        mock_ml_model = Mock(spec=MLModel)
        mock_ml_model.model_type = 'random_forest'
        mock_ml_model.is_trained = True
        
        # Mock predict_single method
        mock_ml_model.predict_single.return_value = {
            'label': 1,
            'probability': 0.85,
            'confidence': 0.7,
            'rationale': 'ML model prediction',
            'latency_ms': 12.5,
            'model_type': 'random_forest'
        }
        
        mock_genai_model = Mock(spec=GenAIBaseline)
        mock_genai_model.config = GenAIConfig(mock_mode=True)
        
        # Mock predict method
        mock_genai_model.predict.return_value = {
            'label': 1,
            'probability': 0.88,
            'confidence': 0.76,
            'rationale': 'GenAI model prediction',
            'latency_ms': 245.3,
            'model_name': 'distilbert',
            'input_features_processed': 30
        }
        
        # Mock batch_predict method
        mock_genai_model.batch_predict.return_value = [
            {
                'label': 1,
                'probability': 0.88,
                'confidence': 0.76,
                'rationale': 'GenAI prediction 1',
                'latency_ms': 245.3,
                'model_name': 'distilbert',
                'input_features_processed': 30
            },
            {
                'label': 0,
                'probability': 0.32,
                'confidence': 0.36,
                'rationale': 'GenAI prediction 2',
                'latency_ms': 210.5,
                'model_name': 'distilbert',
                'input_features_processed': 30
            }
        ]
        
        # Mock preprocessor
        mock_preprocessor = Mock(spec=PreprocessingPipeline)
        mock_preprocessor.transform.return_value = np.random.randn(1, 20)
        
        # Setup mock registry
        mock_registry_instance = Mock()
        mock_registry_instance.ml_model = mock_ml_model
        mock_registry_instance.genai_model = mock_genai_model
        mock_registry_instance.preprocessor = mock_preprocessor
        mock_registry_instance.get_model_status.return_value = {
            'ml_model': {'loaded': True},
            'genai_model': {'loaded': True},
            'preprocessor': {'loaded': True}
        }
        
        mock_registry.return_value = mock_registry_instance
        
        # Create test client
        with TestClient(app) as test_client:
            yield test_client


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["service"] == "EvalOps Lite"
    assert data["version"] == "1.0.0"
    assert "endpoints" in data
    assert "predict" in data["endpoints"]
    assert "metrics" in data["endpoints"]


def test_health_endpoint(client):
    """Test health endpoint"""
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "healthy"
    assert "model_registry" in data
    assert "service" in data
    assert data["service"] == "EvalOps Lite API"


def test_model_status_endpoint(client):
    """Test model status endpoint"""
    response = client.get("/api/v1/models/status")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "available"
    assert "models" in data
    assert "ml_model" in data["models"]
    assert "genai_model" in data["models"]


def test_predict_endpoint(client):
    """Test predict endpoint"""
    # Create sample features
    features = list(np.random.randn(30))
    
    # Make prediction request
    response = client.post("/api/v1/predict", json={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "request_id" in data
    assert "features" in data
    assert "ml_prediction" in data
    assert "genai_prediction" in data
    assert "agreement" in data
    assert "timestamp" in data
    
    # Check predictions
    ml_pred = data["ml_prediction"]
    genai_pred = data["genai_prediction"]
    
    assert ml_pred["label"] in [0, 1]
    assert genai_pred["label"] in [0, 1]
    assert 0 <= ml_pred["probability"] <= 1
    assert 0 <= genai_pred["probability"] <= 1
    
    # Check agreement calculation
    assert data["agreement"] == (ml_pred["label"] == genai_pred["label"])


def test_predict_endpoint_invalid_features(client):
    """Test predict endpoint with invalid features"""
    # Too few features
    response = client.post("/api/v1/predict", json={"features": list(range(29))})
    assert response.status_code == 422  # Validation error
    
    # Too many features
    response = client.post("/api/v1/predict", json={"features": list(range(31))})
    assert response.status_code == 422
    
    # Non-numeric features (JSON would fail parsing, so test with string)
    response = client.post("/api/v1/predict", json={"features": ["a"] * 30})
    # This might pass JSON parsing but fail validation
    if response.status_code != 422:
        # Check if we get a meaningful error
        assert response.status_code >= 400


def test_batch_predict_endpoint(client):
    """Test batch predict endpoint"""
    # Create batch of features
    features_list = [
        list(np.random.randn(30)),
        list(np.random.randn(30)),
        list(np.random.randn(30))
    ]
    
    response = client.post("/api/v1/predict/batch", json={"features_list": features_list})
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 3
    
    for item in data:
        assert "request_id" in item
        assert "ml_prediction" in item
        assert "genai_prediction" in item
        assert "agreement" in item


def test_ml_only_predict_endpoint(client):
    """Test ML-only predict endpoint"""
    features = list(np.random.randn(30))
    
    response = client.get("/api/v1/predict/ml", params={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "label" in data
    assert "probability" in data
    assert "rationale" in data
    assert "latency_ms" in data
    assert data["model_type"] == "random_forest"


def test_genai_only_predict_endpoint(client):
    """Test GenAI-only predict endpoint"""
    features = list(np.random.randn(30))
    
    response = client.get("/api/v1/predict/genai", params={"features": features})
    
    assert response.status_code == 200
    data = response.json()
    
    assert "label" in data
    assert "probability" in data
    assert "rationale" in data
    assert "latency_ms" in data


def test_metrics_endpoint(client, tmp_path):
    """Test metrics endpoint"""
    # First, create a mock metrics file
    metrics_file = tmp_path / "evaluation_results.json"
    
    sample_metrics = [
        {
            "timestamp": "2024-01-01T12:00:00",
            "model_name": "ml_random_forest",
            "eval_type": "comparison",
            "metrics": {
                "accuracy": 0.95,
                "precision": 0.96,
                "recall": 0.94,
                "f1_score": 0.95,
                "inference_latency_p95": 15.2
            }
        }
    ]
    
    metrics_file.write_text(json.dumps(sample_metrics))
    
    # Mock the path in the metrics router
    with patch('src.api.routers.metrics.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = metrics_file
        
        response = client.get("/api/v1/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "latency_metrics" in data
    assert "model_status" in data


def test_metrics_endpoint_no_file(client):
    """Test metrics endpoint when no metrics file exists"""
    # Mock non-existent file
    with patch('src.api.routers.metrics.Path') as mock_path:
        mock_path.return_value.exists.return_value = False
        
        response = client.get("/api/v1/metrics")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "no_metrics_available"
    assert "message" in data


def test_metrics_history_endpoint(client, tmp_path):
    """Test metrics history endpoint"""
    # Create mock metrics file
    metrics_file = tmp_path / "evaluation_results.json"
    
    sample_metrics = [
        {
            "timestamp": f"2024-01-01T12:00:{i:02d}",
            "model_name": f"model_{i % 2}",
            "eval_type": "comparison",
            "metrics": {"accuracy": 0.9 + i * 0.01}
        }
        for i in range(10)
    ]
    
    metrics_file.write_text(json.dumps(sample_metrics))
    
    with patch('src.api.routers.metrics.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = metrics_file
        
        # Test without filter
        response = client.get("/api/v1/metrics/history")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 10
        
        # Test with model filter
        response = client.get("/api/v1/metrics/history?model_name=model_0")
        assert response.status_code == 200
        data = response.json()
        # Should get only model_0 entries
        assert all("model_0" in item["model_name"] for item in data)
        
        # Test with limit
        response = client.get("/api/v1/metrics/history?limit=3")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3


def test_metrics_summary_endpoint(client, tmp_path):
    """Test metrics summary endpoint"""
    # Create mock metrics file
    metrics_file = tmp_path / "evaluation_results.json"
    
    sample_metrics = [
        {
            "timestamp": "2024-01-01T12:00:00",
            "model_name": "ml_model",
            "eval_type": "comparison",
            "metrics": {
                "accuracy": 0.95,
                "f1_score": 0.94,
                "inference_latency_p95": 15.2
            }
        },
        {
            "timestamp": "2024-01-01T12:00:01",
            "model_name": "ml_model",
            "eval_type": "comparison",
            "metrics": {
                "accuracy": 0.92,
                "f1_score": 0.91,
                "inference_latency_p95": 14.8
            }
        }
    ]
    
    metrics_file.write_text(json.dumps(sample_metrics))
    
    with patch('src.api.routers.metrics.Path') as mock_path:
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.__truediv__.return_value = metrics_file
        
        response = client.get("/api/v1/metrics/summary")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert "summary_by_model" in data
    assert "ml_model" in data["summary_by_model"]
    assert "accuracy" in data["summary_by_model"]["ml_model"]


def test_error_handling(client):
    """Test error handling in API"""
    # Test with broken model (simulate error)
    with patch('src.api.routers.predict.get_ml_model') as mock_get_ml:
        mock_get_ml.side_effect = Exception("Model loading failed")
        
        features = list(np.random.randn(30))
        response = client.post("/api/v1/predict", json={"features": features})
        
        # Should return 503 or 500 error
        assert response.status_code >= 500
        error_data = response.json()
        assert "error" in error_data
        assert "detail" in error_data


def test_cors_headers(client):
    """Test CORS headers are present"""
    response = client.get("/")
    
    # Check for CORS headers
    assert "access-control-allow-origin" in response.headers
    assert response.headers["access-control-allow-origin"] == "*"
    
    # OPTIONS request should also work
    response = client.options("/")
    assert response.status_code == 200  # or 204 depending on implementation


def test_api_versioning():
    """Test API versioning in routes"""
    app = create_app()
    
    # Check that routes have /api/v1 prefix
    routes = [route.path for route in app.routes]
    
    assert "/api/v1/predict" in routes
    assert "/api/v1/metrics" in routes
    assert "/api/v1/models/status" in routes


def test_global_exception_handler(client):
    """Test global exception handler"""
    # Mock an endpoint to raise an exception
    original_predict = client.app.router.routes[0].endpoint
    
    async def failing_endpoint():
        raise ValueError("Test exception")
    
    # Temporarily replace an endpoint
    client.app.router.routes[0].endpoint = failing_endpoint
    
    try:
        response = client.get("/")
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "detail" in data
        assert "Internal server error" in data["error"]
    finally:
        # Restore original endpoint
        client.app.router.routes[0].endpoint = original_predict
