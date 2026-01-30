"""
Tests for GenAI baseline module
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.models.genai_baseline import GenAIBaseline, GenAIConfig
from src.data.schemas import GenAIOutputSchema


def test_genai_config_defaults():
    """Test GenAI configuration defaults"""
    config = GenAIConfig()
    
    assert config.model_name == "distilbert-base-uncased"
    assert config.device == "cpu"
    assert config.max_length == 512
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.include_rationale == True
    assert config.mock_mode == False


def test_genai_baseline_initialization():
    """Test GenAI baseline initialization"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    
    assert model.config == config
    assert model.model is None
    assert model.tokenizer is None
    assert model.classifier is None
    assert not model.is_loaded
    
    # Check templates exist
    assert len(model.feature_templates) > 0
    assert 0 in model.rationale_templates
    assert 1 in model.rationale_templates


def test_genai_baseline_load_model():
    """Test loading GenAI model"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    
    # Load model in mock mode
    model.load_model()
    
    assert model.is_loaded
    assert model.config.mock_mode == True


@patch('src.models.genai_baseline.AutoTokenizer.from_pretrained')
@patch('src.models.genai_baseline.AutoModelForSequenceClassification.from_pretrained')
def test_genai_baseline_load_real_model(mock_model, mock_tokenizer):
    """Test loading real transformer model (mocked)"""
    # Mock the transformers
    mock_tokenizer.return_value = Mock()
    mock_model_instance = Mock()
    mock_model.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance
    
    config = GenAIConfig(mock_mode=False)
    model = GenAIBaseline(config)
    
    # Mock pipeline
    with patch('src.models.genai_baseline.pipeline') as mock_pipeline:
        mock_pipeline.return_value = Mock()
        
        # Load model
        model.load_model()
        
        assert model.is_loaded
        assert not model.config.mock_mode
        mock_tokenizer.assert_called_once_with(config.model_name)
        mock_model.assert_called_once()


def test_features_to_text():
    """Test converting features to descriptive text"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    
    # Create sample features (30 values)
    features = list(range(30))
    
    text = model._features_to_text(features)
    
    # Check text generation
    assert isinstance(text, str)
    assert len(text) > 0
    
    # Should include feature values
    for i in range(10):  # First 10 features are used
        assert str(features[i]) in text or f"{features[i]:.2f}" in text
    
    # Should include analysis instruction
    assert "Analyze" in text or "analysis" in text.lower()


def test_mock_predict():
    """Test mock prediction"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    
    # Test with features in text
    text_with_features = "features: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]"
    
    label, probability, rationale = model._mock_predict(text_with_features)
    
    assert label in [0, 1]
    assert 0 <= probability <= 1
    assert isinstance(rationale, str)
    assert len(rationale) > 0
    
    # Test without features
    text_without_features = "Sample medical text for analysis"
    
    label2, probability2, rationale2 = model._mock_predict(text_without_features)
    
    assert label2 in [0, 1]
    assert 0 <= probability2 <= 1
    assert isinstance(rationale2, str)


def test_genai_baseline_predict():
    """Test GenAI baseline prediction"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    model.load_model()
    
    # Create sample features
    features = list(np.random.randn(30))
    
    # Make prediction
    result = model.predict(features)
    
    # Check result structure
    assert 'label' in result
    assert 'probability' in result
    assert 'confidence' in result
    assert 'rationale' in result
    assert 'latency_ms' in result
    assert 'model_name' in result
    assert 'input_features_processed' in result
    
    # Check value ranges
    assert result['label'] in [0, 1]
    assert 0 <= result['probability'] <= 1
    assert 0 <= result['confidence'] <= 1
    assert result['latency_ms'] > 0
    assert result['input_features_processed'] == 30
    
    # Validate against schema
    try:
        GenAIOutputSchema(**result)
        validation_passed = True
    except:
        validation_passed = False
    
    assert validation_passed, "Result should pass schema validation"


def test_genai_baseline_batch_predict():
    """Test batch prediction"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    model.load_model()
    
    # Create batch of features
    features_list = [
        list(np.random.randn(30)) for _ in range(5)
    ]
    
    # Make batch predictions
    results = model.batch_predict(features_list)
    
    # Check results
    assert len(results) == 5
    
    for result in results:
        assert 'label' in result
        assert 'probability' in result
        assert 'rationale' in result
        assert result['label'] in [0, 1]
        assert 0 <= result['probability'] <= 1


def test_genai_baseline_get_config():
    """Test getting model configuration"""
    config = GenAIConfig(
        model_name="test-model",
        device="cuda",
        max_length=256,
        temperature=0.5,
        mock_mode=True
    )
    
    model = GenAIBaseline(config)
    model.load_model()
    
    config_dict = model.get_config()
    
    assert config_dict['model_name'] == "test-model"
    assert config_dict['device'] == "cuda"
    assert config_dict['max_length'] == 256
    assert config_dict['temperature'] == 0.5
    assert config_dict['mock_mode'] == True
    assert config_dict['is_loaded'] == True


def test_prediction_with_invalid_features():
    """Test prediction with invalid features"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    model.load_model()
    
    # Test with wrong number of features
    invalid_features = list(range(29))  # Should be 30
    
    # Should raise validation error
    with pytest.raises(ValueError):
        model.predict(invalid_features)


def test_rationale_generation():
    """Test rationale generation based on prediction"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    
    # Test rationale templates
    assert 0 in model.rationale_templates
    assert 1 in model.rationale_templates
    
    # Check template formatting
    template_0 = model.rationale_templates[0]
    template_1 = model.rationale_templates[1]
    
    # Should contain placeholder for confidence
    assert "{confidence:" in template_0 or "{confidence" in template_0
    assert "{confidence:" in template_1 or "{confidence" in template_1


def test_prediction_error_handling():
    """Test error handling in prediction"""
    config = GenAIConfig(mock_mode=True)
    model = GenAIBaseline(config)
    model.load_model()
    
    # Mock an error in _mock_predict
    original_mock_predict = model._mock_predict
    
    def raising_mock_predict(text):
        raise Exception("Simulated prediction error")
    
    model._mock_predict = raising_mock_predict
    
    # Should return safe default
    features = list(np.random.randn(30))
    result = model.predict(features)
    
    # Should still return valid structure
    assert 'label' in result
    assert 'probability' in result
    assert 'rationale' in result
    assert 'error' in result
    
    # Check safe defaults
    assert result['label'] == 0
    assert result['probability'] == 0.5
    assert result['confidence'] == 0.0
    
    # Restore original method
    model._mock_predict = original_mock_predict
