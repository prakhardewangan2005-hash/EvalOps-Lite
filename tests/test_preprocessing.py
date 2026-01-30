"""
Tests for preprocessing module
"""

import pytest
import numpy as np
from src.preprocessing.transformers import FeatureScaler, OutlierHandler, FeatureSelector
from src.preprocessing.pipeline import PreprocessingPipeline


def test_feature_scaler_initialization():
    """Test FeatureScaler initialization"""
    # Test standard scaler
    scaler_standard = FeatureScaler(scaling_strategy='standard')
    assert scaler_standard.scaling_strategy == 'standard'
    
    # Test robust scaler
    scaler_robust = FeatureScaler(scaling_strategy='robust')
    assert scaler_robust.scaling_strategy == 'robust'
    
    # Test invalid strategy
    with pytest.raises(ValueError):
        FeatureScaler(scaling_strategy='invalid')


def test_feature_scaler_fit_transform():
    """Test FeatureScaler fit and transform"""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 5) * 10 + 5
    
    # Test standard scaler
    scaler = FeatureScaler(scaling_strategy='standard')
    X_scaled = scaler.fit_transform(X)
    
    # Check that mean is ~0 and std is ~1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-10)
    
    # Test robust scaler
    scaler_robust = FeatureScaler(scaling_strategy='robust')
    X_scaled_robust = scaler_robust.fit_transform(X)
    
    # Robust scaler should handle outliers better
    assert X_scaled_robust.shape == X.shape


def test_outlier_handler_initialization():
    """Test OutlierHandler initialization"""
    handler = OutlierHandler(threshold=3.0)
    assert handler.threshold == 3.0
    assert handler.lower_bounds is None
    assert handler.upper_bounds is None


def test_outlier_handler_fit_transform():
    """Test OutlierHandler fit and transform"""
    np.random.seed(42)
    
    # Create data with outliers
    X = np.random.randn(100, 3)
    
    # Add some outliers
    X[0, 0] = 100  # Extreme positive outlier
    X[1, 1] = -100  # Extreme negative outlier
    
    handler = OutlierHandler(threshold=3.0)
    handler.fit(X)
    
    # Check bounds are calculated
    assert handler.lower_bounds is not None
    assert handler.upper_bounds is not None
    assert len(handler.lower_bounds) == 3
    assert len(handler.upper_bounds) == 3
    
    # Transform data
    X_transformed = handler.transform(X)
    
    # Check outliers are clipped
    assert X_transformed[0, 0] <= handler.upper_bounds[0]
    assert X_transformed[1, 1] >= handler.lower_bounds[1]
    
    # Check non-outliers unchanged
    assert np.allclose(X_transformed[2:, :], X[2:, :])


def test_feature_selector_initialization():
    """Test FeatureSelector initialization"""
    selector = FeatureSelector(n_features=10)
    assert selector.n_features == 10
    assert selector.selected_indices is None


def test_feature_selector_fit_transform():
    """Test FeatureSelector fit and transform"""
    np.random.seed(42)
    
    # Create data with varying variances
    X = np.column_stack([
        np.random.randn(100) * 1.0,    # Low variance
        np.random.randn(100) * 5.0,    # Medium variance
        np.random.randn(100) * 10.0,   # High variance
        np.random.randn(100) * 20.0,   # Very high variance
        np.random.randn(100) * 0.5     # Very low variance
    ])
    
    selector = FeatureSelector(n_features=3)
    selector.fit(X)
    
    # Check selected indices
    assert selector.selected_indices is not None
    assert len(selector.selected_indices) == 3
    
    # Highest variance features should be selected
    variances = np.var(X, axis=0)
    expected_indices = np.argsort(variances)[-3:]
    
    assert set(selector.selected_indices) == set(expected_indices)
    
    # Transform data
    X_selected = selector.transform(X)
    
    # Check shape
    assert X_selected.shape == (100, 3)


def test_preprocessing_pipeline_initialization():
    """Test PreprocessingPipeline initialization"""
    pipeline = PreprocessingPipeline()
    
    assert pipeline.config is not None
    assert 'imputation_strategy' in pipeline.config
    assert 'scaling_strategy' in pipeline.config
    assert pipeline.pipeline is None
    assert not pipeline.is_fitted


def test_preprocessing_pipeline_build():
    """Test building preprocessing pipeline"""
    pipeline = PreprocessingPipeline()
    built_pipeline = pipeline.build_pipeline()
    
    assert built_pipeline is not None
    assert hasattr(built_pipeline, 'steps')
    assert len(built_pipeline.steps) == 4
    assert built_pipeline.steps[0][0] == 'imputer'
    assert built_pipeline.steps[1][0] == 'outlier_handler'
    assert built_pipeline.steps[2][0] == 'feature_selector'
    assert built_pipeline.steps[3][0] == 'scaler'


def test_preprocessing_pipeline_fit_transform():
    """Test pipeline fit and transform"""
    np.random.seed(42)
    X = np.random.randn(100, 30)
    
    pipeline = PreprocessingPipeline()
    X_transformed = pipeline.fit_transform(X)
    
    # Check pipeline is fitted
    assert pipeline.is_fitted
    
    # Check transformed shape
    assert X_transformed.shape[0] == 100
    assert X_transformed.shape[1] == 20  # Default n_features in config
    
    # Check no NaN values
    assert not np.any(np.isnan(X_transformed))
    
    # Test transform separately
    X_test = np.random.randn(50, 30)
    X_test_transformed = pipeline.transform(X_test)
    
    assert X_test_transformed.shape == (50, 20)


def test_preprocessing_pipeline_save_load(tmp_path):
    """Test saving and loading pipeline"""
    np.random.seed(42)
    X = np.random.randn(100, 30)
    
    # Create and fit pipeline
    pipeline1 = PreprocessingPipeline()
    pipeline1.fit(X)
    
    # Save pipeline
    save_path = tmp_path / "pipeline.pkl"
    pipeline1.save(str(save_path))
    
    # Create new pipeline and load
    pipeline2 = PreprocessingPipeline()
    pipeline2.load(str(save_path))
    
    # Check loaded pipeline is fitted
    assert pipeline2.is_fitted
    
    # Test transform with loaded pipeline
    X_test = np.random.randn(50, 30)
    X_transformed1 = pipeline1.transform(X_test)
    X_transformed2 = pipeline2.transform(X_test)
    
    # Results should be identical
    assert np.allclose(X_transformed1, X_transformed2)


def test_get_feature_names():
    """Test getting feature names after selection"""
    np.random.seed(42)
    X = np.random.randn(100, 30)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(30)]
    
    # Create and fit pipeline
    pipeline = PreprocessingPipeline()
    pipeline.fit(X)
    
    # Get selected feature names
    selected_names = pipeline.get_feature_names(feature_names)
    
    # Check we get correct number of names
    assert len(selected_names) == 20  # Default n_features
    
    # Check all names are from original list
    for name in selected_names:
        assert name in feature_names
