"""
Tests for data ingestion module
"""

import pytest
import numpy as np
from src.data.ingestion import DataIngestor
from src.data.schemas import DataSchema, PredictionRequestSchema


def test_data_ingestor_initialization():
    """Test DataIngestor initialization"""
    ingestor = DataIngestor(test_size=0.2, random_state=42)
    
    assert ingestor.test_size == 0.2
    assert ingestor.random_state == 42
    assert ingestor.dataset is None


def test_load_breast_cancer_data():
    """Test loading breast cancer dataset"""
    ingestor = DataIngestor(test_size=0.3, random_state=42)
    dataset = ingestor.load_breast_cancer_data()
    
    # Check dataset structure
    assert dataset.X_train.shape[0] > 0
    assert dataset.X_test.shape[0] > 0
    assert dataset.y_train.shape[0] == dataset.X_train.shape[0]
    assert dataset.y_test.shape[0] == dataset.X_test.shape[0]
    
    # Check feature names
    assert len(dataset.feature_names) == 30
    assert 'mean radius' in dataset.feature_names
    assert 'mean texture' in dataset.feature_names
    
    # Check target names
    assert len(dataset.target_names) == 2
    assert 'malignant' in dataset.target_names or 'benign' in dataset.target_names
    
    # Check metadata
    assert 'dataset_name' in dataset.metadata
    assert dataset.metadata['dataset_name'] == 'breast_cancer'
    assert dataset.metadata['n_features'] == 30
    assert dataset.metadata['n_classes'] == 2


def test_get_feature_statistics():
    """Test feature statistics calculation"""
    ingestor = DataIngestor()
    dataset = ingestor.load_breast_cancer_data()
    stats = ingestor.get_feature_statistics()
    
    assert 'training_samples' in stats
    assert 'test_samples' in stats
    assert 'n_features' in stats
    assert stats['n_features'] == 30
    assert 'feature_names' in stats
    assert len(stats['feature_names']) == 30


def test_data_schema_validation():
    """Test data schema validation"""
    schema = DataSchema()
    
    # Create valid dataframe
    import pandas as pd
    import numpy as np
    
    # Generate synthetic valid data
    n_samples = 100
    n_features = 30
    data = np.random.randn(n_samples, n_features)
    
    # Create column names matching breast cancer dataset
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    df['target'] = np.random.randint(0, 2, n_samples)
    
    # Should pass validation
    assert schema.validate(df) == True
    
    # Test with missing column
    df_invalid = df.drop(columns=['mean radius'])
    with pytest.raises(ValueError):
        schema.validate(df_invalid)
    
    # Test with invalid target values
    df_invalid_target = df.copy()
    df_invalid_target['target'] = 2
    with pytest.raises(ValueError):
        schema.validate(df_invalid_target)


def test_prediction_request_schema():
    """Test prediction request schema validation"""
    # Valid request
    valid_features = list(range(30))
    schema = PredictionRequestSchema(features=valid_features)
    assert len(schema.features) == 30
    
    # Invalid request - wrong length
    with pytest.raises(ValueError):
        PredictionRequestSchema(features=list(range(29)))
    
    with pytest.raises(ValueError):
        PredictionRequestSchema(features=list(range(31)))


def test_dataset_class_distribution():
    """Test that dataset has reasonable class distribution"""
    ingestor = DataIngestor(test_size=0.2, random_state=42)
    dataset = ingestor.load_breast_cancer_data()
    
    # Check that we have samples from both classes
    unique_train = np.unique(dataset.y_train)
    unique_test = np.unique(dataset.y_test)
    
    assert len(unique_train) == 2
    assert len(unique_test) == 2
    
    # Check stratification (roughly similar distribution)
    train_dist = np.bincount(dataset.y_train)
    test_dist = np.bincount(dataset.y_test)
    
    train_ratio = train_dist[1] / train_dist.sum()
    test_ratio = test_dist[1] / test_dist.sum()
    
    # Should be roughly similar due to stratification
    assert abs(train_ratio - test_ratio) < 0.1
