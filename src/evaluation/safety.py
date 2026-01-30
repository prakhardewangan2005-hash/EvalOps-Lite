"""
Safety checks and validation for model outputs
"""

from typing import Dict, Any, List, Tuple
import logging
from dataclasses import dataclass

from src.data.schemas import GenAIOutputSchema

logger = logging.getLogger(__name__)


@dataclass
class SafetyCheckResult:
    """Result of safety check"""
    passed: bool
    violations: List[str]
    score: float  # 0.0 to 1.0


class SafetyValidator:
    """
    Validates model outputs for safety and compliance
    """
    
    def __init__(self):
        """Initialize safety validator"""
        self.violation_history: List[Dict[str, Any]] = []
        
    def validate_genai_output(self, output: Dict[str, Any]) -> SafetyCheckResult:
        """
        Validate GenAI output against schema and safety rules
        
        Args:
            output: GenAI output dictionary
            
        Returns:
            Safety check result
        """
        violations = []
        
        # 1. Schema validation
        try:
            GenAIOutputSchema(**output)
        except Exception as e:
            violations.append(f"Schema violation: {str(e)}")
        
        # 2. Label safety check
        if 'label' in output:
            if output['label'] not in [0, 1]:
                violations.append(f"Invalid label: {output['label']}")
        
        # 3. Probability confidence check
        if 'probability' in output and 'confidence' in output:
            prob = output['probability']
            conf = output['confidence']
            
            # Check for overconfidence in uncertain predictions
            if 0.4 <= prob <= 0.6 and conf > 0.8:
                violations.append("Overconfident in uncertain prediction")
            
            # Check for inconsistency between probability and confidence
            expected_confidence = abs(prob - 0.5) * 2
            if abs(conf - expected_confidence) > 0.3:
                violations.append(f"Confidence-probability mismatch: {conf:.2f} vs expected {expected_confidence:.2f}")
        
        # 4. Rationale safety check
        if 'rationale' in output:
            rationale = output['rationale'].lower()
            
            # Check for harmful language (basic check)
            harmful_terms = ['definitely', 'certainly', 'guarantee', 'always', 'never']
            for term in harmful_terms:
                if term in rationale and 'not ' + term not in rationale:
                    violations.append(f"Overly certain language: '{term}'")
            
            # Check rationale length
            if len(output['rationale']) < 20:
                violations.append("Rationale too short (minimum 20 characters)")
            elif len(output['rationale']) > 500:
                violations.append("Rationale too long (maximum 500 characters)")
        
        # 5. Latency check
        if 'latency_ms' in output:
            latency = output['latency_ms']
            if latency > 10000:  # 10 seconds
                violations.append(f"Excessive latency: {latency:.0f}ms")
            elif latency < 1:  # Unrealistically fast
                violations.append(f"Unrealistically fast: {latency:.0f}ms")
        
        # Calculate safety score
        passed = len(violations) == 0
        score = max(0.0, 1.0 - (len(violations) * 0.2))  # Deduct 0.2 per violation
        
        result = SafetyCheckResult(
            passed=passed,
            violations=violations,
            score=score
        )
        
        # Log violations
        if violations:
            self.violation_history.append({
                'output': output,
                'violations': violations,
                'score': score
            })
            logger.warning(f"Safety violations: {violations}")
        
        return result
    
    def validate_batch(self, outputs: List[Dict[str, Any]]) -> List[SafetyCheckResult]:
        """
        Validate batch of GenAI outputs
        
        Args:
            outputs: List of GenAI output dictionaries
            
        Returns:
            List of safety check results
        """
        return [self.validate_genai_output(output) for output in outputs]
    
    def get_safety_report(self) -> Dict[str, Any]:
        """
        Generate safety report from validation history
        
        Returns:
            Safety report dictionary
        """
        if not self.violation_history:
            return {
                'total_checks': 0,
                'passed_checks': 0,
                'violation_rate': 0.0,
                'average_score': 1.0,
                'common_violations': {}
            }
        
        total_checks = len(self.violation_history)
        passed_checks = sum(1 for item in self.violation_history if item['score'] == 1.0)
        
        # Count common violations
        violation_counts = {}
        for item in self.violation_history:
            for violation in item['violations']:
                violation_type = violation.split(':')[0] if ':' in violation else violation
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
        
        # Calculate average score
        average_score = sum(item['score'] for item in self.violation_history) / total_checks
        
        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'violation_rate': 1.0 - (passed_checks / total_checks),
            'average_score': average_score,
            'common_violations': violation_counts
        }
    
    def clear_history(self):
        """Clear validation history"""
        self.violation_history.clear()
