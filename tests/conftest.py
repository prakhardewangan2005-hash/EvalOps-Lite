"""
Pytest configuration and fixtures
"""

import pytest
import numpy as np
from typing import Generator, Tuple
import tempfile
import shutil
from pathlib import Path

from src.data.ingestion import DataIngestor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline, GenAIConfig
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.safety import SafetyValidator


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sample dataset for testing"""
    ingestor = DataIngestor(test_size=0.3, random_state=42)
    dataset = ingestor.load_breast_cancer_data()
    
    return dataset.X_train, dataset.X_test, dataset.y_train, dataset.y_test


@pytest.fixture
def trained_ml_model(sample_dataset) -> MLModel:
    """Create and train an ML model for testing"""
    X_train, _, y_train, _ = sample_dataset
    
    # Create preprocessing pipeline
    preprocessor = PreprocessingPipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Train ML model
    model = MLModel(model_type='random_forest', n_estimators=10, max_depth=5)
    model.train(X_train_processed, y_train)
    
    return model


@pytest.fixture
def genai_model() -> GenAIBaseline:
    """Create GenAI model for testing"""
    config = GenAIConfig(mock_mode=True)  # Use mock for testing
    model = GenAIBaseline(config)
    model.load_model()
    
    return model


@pytest.fixture
def preprocessor(sample_dataset) -> PreprocessingPipeline:
    """Create fitted preprocessor for testing"""
    X_train, _, _, _ = sample_dataset
    
    preprocessor = PreprocessingPipeline()
    preprocessor.fit(X_train)
    
    return preprocessor


@pytest.fixture
def model_evaluator() -> ModelEvaluator:
    """Create model evaluator for testing"""
    return ModelEvaluator()


@pytest.fixture
def safety_validator() -> SafetyValidator:
    """Create safety validator for testing"""
    return SafetyValidator()


@pytest.fixture
def sample_features() -> list:
    """Generate sample features for testing"""
    return [
        17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
        1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
        25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]


@pytest.fixture
def sample_genai_output() -> dict:
    """Generate sample GenAI output for testing"""
    return {
        'label': 1,
        'probability': 0.85,
        'confidence': 0.7,
        'rationale': 'Based on feature analysis, this appears to be malignant with high confidence.',
        'latency_ms': 45.2,
        'model_name': 'distilbert-base-uncased'
    }


@pytest.fixture
def artifacts_dir(temp_dir) -> Path:
    """Create artifacts directory for testing"""
    artifacts_path = Path(temp_dir) / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    return artifacts_path
