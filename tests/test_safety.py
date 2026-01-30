"""
Tests for safety validation module
"""

import pytest
from src.evaluation.safety import SafetyValidator, SafetyCheckResult
from src.data.schemas import GenAIOutputSchema


def test_safety_check_result_dataclass():
    """Test SafetyCheckResult dataclass"""
    result = SafetyCheckResult(
        passed=True,
        violations=['violation1', 'violation2'],
        score=0.8
    )
    
    assert result.passed == True
    assert result.violations == ['violation1', 'violation2']
    assert result.score == 0.8


def test_safety_validator_initialization():
    """Test SafetyValidator initialization"""
    validator = SafetyValidator()
    
    assert validator.violation_history == []
    assert len(validator.violation_history) == 0


def test_validate_genai_output_schema():
    """Test schema validation in safety check"""
    validator = SafetyValidator()
    
    # Valid output
    valid_output = {
        'label': 0,
        'probability': 0.3,
        'confidence': 0.4,
        'rationale': 'A sufficiently long rationale that meets the minimum length requirement.',
        'latency_ms': 25.5
    }
    
    result = validator.validate_genai_output(valid_output)
    
    assert result.passed == True
    assert len(result.violations) == 0
    assert result.score == 1.0
    
    # Also validate with schema directly
    schema = GenAIOutputSchema(**valid_output)
    assert schema.label == 0
    assert schema.probability == 0.3


def test_validate_genai_output_label_safety():
    """Test label safety validation"""
    validator = SafetyValidator()
    
    # Invalid label
    invalid_output = {
        'label': 2,  # Should be 0 or 1
        'probability': 0.5,
        'confidence': 0.5,
        'rationale': 'Valid rationale length here.',
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(invalid_output)
    
    assert result.passed == False
    assert any('label' in v.lower() for v in result.violations)
    assert result.score < 1.0


def test_validate_genai_output_probability_confidence():
    """Test probability and confidence validation"""
    validator = SafetyValidator()
    
    # Overconfident in uncertain prediction
    overconfident_output = {
        'label': 0,
        'probability': 0.45,  # Near 0.5
        'confidence': 0.9,    # High confidence despite uncertainty
        'rationale': 'Valid rationale length here.',
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(overconfident_output)
    
    assert result.passed == False
    assert any('overconfident' in v.lower() for v in result.violations)
    
    # Probability-confidence mismatch
    mismatch_output = {
        'label': 1,
        'probability': 0.9,   # High probability
        'confidence': 0.2,    # Low confidence (should be high)
        'rationale': 'Valid rationale length here.',
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(mismatch_output)
    
    assert result.passed == False
    assert any('mismatch' in v.lower() for v in result.violations)


def test_validate_genai_output_rationale_safety():
    """Test rationale safety validation"""
    validator = SafetyValidator()
    
    # Rationale too short
    short_output = {
        'label': 1,
        'probability': 0.8,
        'confidence': 0.6,
        'rationale': 'short',  # Too short
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(short_output)
    
    assert result.passed == False
    assert any('too short' in v.lower() for v in result.violations)
    
    # Rationale too long
    long_output = {
        'label': 1,
        'probability': 0.8,
        'confidence': 0.6,
        'rationale': 'x' * 600,  # Too long
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(long_output)
    
    assert result.passed == False
    assert any('too long' in v.lower() for v in result.violations)
    
    # Harmful language
    harmful_output = {
        'label': 1,
        'probability': 0.8,
        'confidence': 0.6,
        'rationale': 'This is definitely malignant, I guarantee it.',
        'latency_ms': 10.0
    }
    
    result = validator.validate_genai_output(harmful_output)
    
    assert result.passed == False
    assert any('certain' in v.lower() for v in result.violations)


def test_validate_genai_output_latency_check():
    """Test latency safety validation"""
    validator = SafetyValidator()
    
    # Excessive latency
    slow_output = {
        'label': 1,
        'probability': 0.8,
        'confidence': 0.6,
        'rationale': 'Valid rationale length here.',
        'latency_ms': 15000.0  # 15 seconds
    }
    
    result = validator.validate_genai_output(slow_output)
    
    assert result.passed == False
    assert any('excessive' in v.lower() or 'latency' in v.lower() for v in result.violations)
    
    # Unrealistically fast
    fast_output = {
        'label': 1,
        'probability': 0.8,
        'confidence': 0.6,
        'rationale': 'Valid rationale length here.',
        'latency_ms': 0.1  # 0.1ms
    }
    
    result = validator.validate_genai_output(fast_output)
    
    assert result.passed == False
    assert any('fast' in v.lower() or 'unrealistically' in v.lower() for v in result.violations)


def test_validate_batch():
    """Test batch validation"""
    validator = SafetyValidator()
    
    outputs = [
        {
            'label': 0,
            'probability': 0.3,
            'confidence': 0.4,
            'rationale': 'Valid rationale length here.',
            'latency_ms': 25.5
        },
        {
            'label': 2,  # Invalid
            'probability': 1.5,  # Invalid
            'confidence': 0.5,
            'rationale': 'short',  # Too short
            'latency_ms': 10.0
        }
    ]
    
    results = validator.validate_batch(outputs)
    
    assert len(results) == 2
    assert results[0].passed == True
    assert results[1].passed == False
    assert len(results[1].violations) > 0


def test_get_safety_report():
    """Test safety report generation"""
    validator = SafetyValidator()
    
    # Add some validation results
    outputs = [
        {
            'label': 0,
            'probability': 0.3,
            'confidence': 0.4,
            'rationale': 'Valid rationale.',
            'latency_ms': 25.5
        },
        {
            'label': 2,
            'probability': 1.5,
            'confidence': 0.5,
            'rationale': 'short',
            'latency_ms': 10.0
        }
    ]
    
    # Validate outputs
    for output in outputs:
        validator.validate_genai_output(output)
    
    # Get report
    report = validator.get_safety_report()
    
    assert 'total_checks' in report
    assert 'passed_checks' in report
    assert 'violation_rate' in report
    assert 'average_score' in report
    assert 'common_violations' in report
    
    assert report['total_checks'] == 2
    assert report['passed_checks'] == 1
    assert report['violation_rate'] == 0.5
    assert 0 <= report['average_score'] <= 1
    assert isinstance(report['common_violations'], dict)


def test_get_safety_report_empty():
    """Test safety report with no validations"""
    validator = SafetyValidator()
    
    report = validator.get_safety_report()
    
    assert report['total_checks'] == 0
    assert report['passed_checks'] == 0
    assert report['violation_rate'] == 0.0
    assert report['average_score'] == 1.0
    assert report['common_violations'] == {}


def test_clear_history():
    """Test clearing validation history"""
    validator = SafetyValidator()
    
    # Add some validations
    output = {
        'label': 0,
        'probability': 0.3,
        'confidence': 0.4,
        'rationale': 'Valid rationale.',
        'latency_ms': 25.5
    }
    
    validator.validate_genai_output(output)
    
    assert len(validator.violation_history) == 1
    
    # Clear history
    validator.clear_history()
    
    assert len(validator.violation_history) == 0


def test_safety_score_calculation():
    """Test safety score calculation"""
    validator = SafetyValidator()
    
    # Perfect output
    perfect_output = {
        'label': 0,
        'probability': 0.3,
        'confidence': 0.4,
        'rationale': 'Valid rationale of sufficient length.',
        'latency_ms': 25.5
    }
    
    perfect_result = validator.validate_genai_output(perfect_output)
    assert perfect_result.score == 1.0
    
    # Output with one violation
    one_violation_output = {
        'label': 2,  # Violation
        'probability': 0.5,
        'confidence': 0.5,
        'rationale': 'Valid rationale of sufficient length.',
        'latency_ms': 25.5
    }
    
    one_violation_result = validator.validate_genai_output(one_violation_output)
    assert one_violation_result.score == 0.8  # 1.0 - 0.2
    
    # Output with multiple violations
    multi_violation_output = {
        'label': 2,  # Violation
        'probability': 1.5,  # Violation
        'confidence': -0.1,  # Violation
        'rationale': 'short',  # Violation
        'latency_ms': 0.1  # Violation
    }
    
    multi_violation_result = validator.validate_genai_output(multi_violation_output)
    assert multi_violation_result.score == 0.0  # 1.0 - (5 * 0.2)


def test_validate_genai_output_missing_fields():
    """Test validation with missing fields"""
    validator = SafetyValidator()
    
    # Missing required fields
    incomplete_output = {
        'label': 1,
        'probability': 0.8
        # Missing confidence, rationale, latency_ms
    }
    
    result = validator.validate_genai_output(incomplete_output)
    
    assert result.passed == False
    assert len(result.violations) > 0
    assert result.score < 1.0
