"""
Comparator for ML vs GenAI model evaluation
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import json
from datetime import datetime
from pathlib import Path

from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.safety import SafetyValidator
from src.evaluation.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compares ML and GenAI models comprehensively
    """
    
    def __init__(self, output_dir: str = "artifacts"):
        """
        Initialize model comparator
        
        Args:
            output_dir: Directory for output artifacts
        """
        self.evaluator = ModelEvaluator()
        self.safety_validator = SafetyValidator()
        self.metrics_tracker = MetricsTracker(output_dir)
        self.comparison_history: List[Dict[str, Any]] = []
        
    def compare_models(
        self,
        ml_model: Any,
        genai_model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        run_safety_checks: bool = True,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Compare ML and GenAI models
        
        Args:
            ml_model: ML model instance
            genai_model: GenAI model instance
            X_test: Test features
            y_test: Test labels
            run_safety_checks: Whether to run safety checks
            save_results: Whether to save results to disk
            
        Returns:
            Comparison results dictionary
        """
        logger.info("Starting model comparison...")
        
        # Evaluate ML model
        ml_results = self.evaluator.evaluate_ml_model(
            ml_model, X_test, y_test,
            calculate_latency=True
        )
        
        # Evaluate GenAI model
        genai_results = self.evaluator.evaluate_genai_model(
            genai_model, X_test, y_test,
            calculate_latency=True
        )
        
        # Run safety checks on GenAI outputs if requested
        safety_report = None
        if run_safety_checks:
            # Get predictions for safety validation
            X_test_list = X_test.tolist()
            genai_predictions = genai_model.batch_predict(X_test_list[:100])  # Sample 100 for safety
            
            safety_results = self.safety_validator.validate_batch(genai_predictions)
            safety_report = self.safety_validator.get_safety_report()
            
            genai_results['safety_report'] = safety_report
        
        # Track metrics
        self.metrics_tracker.add_metrics(
            ml_results['metrics'], 
            f"ml_{ml_model.model_type}", 
            "comparison"
        )
        self.metrics_tracker.add_metrics(
            genai_results['metrics'], 
            f"genai_{genai_model.config.model_name}", 
            "comparison"
        )
        
        # Perform comparison
        comparison = self._perform_comparison(ml_results, genai_results)
        
        # Prepare results
        results = {
            'comparison_id': f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'test_set_size': X_test.shape[0],
            'ml_model': {
                'type': ml_model.model_type,
                'results': ml_results
            },
            'genai_model': {
                'type': genai_model.config.model_name,
                'results': genai_results
            },
            'comparison': comparison,
            'safety_report': safety_report,
            'winner': self._determine_winner(comparison, safety_report)
        }
        
        self.comparison_history.append(results)
        
        # Save results if requested
        if save_results:
            self._save_comparison_results(results)
            self.metrics_tracker.save_metrics()
        
        logger.info("Model comparison completed")
        logger.info(f"ML Accuracy: {ml_results['metrics']['accuracy']:.4f}")
        logger.info(f"GenAI Accuracy: {genai_results['metrics']['accuracy']:.4f}")
        logger.info(f"Winner: {results['winner']}")
        
        return results
    
    def _perform_comparison(
        self, 
        ml_results: Dict[str, Any], 
        genai_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform detailed comparison between models
        
        Args:
            ml_results: ML model results
            genai_results: GenAI model results
            
        Returns:
            Comparison metrics
        """
        ml_metrics = ml_results['metrics']
        genai_metrics = genai_results['metrics']
        
        comparison = {
            'accuracy_difference': genai_metrics['accuracy'] - ml_metrics['accuracy'],
            'f1_difference': genai_metrics['f1_score'] - ml_metrics['f1_score'],
            'latency_ratio': genai_metrics.get('inference_latency_p95', 100) / 
                           ml_metrics.get('inference_latency_p95', 1),
            'performance_score': self._calculate_performance_score(ml_metrics, genai_metrics),
            'key_metrics': {
                'ml_accuracy': ml_metrics['accuracy'],
                'genai_accuracy': genai_metrics['accuracy'],
                'ml_precision': ml_metrics['precision'],
                'genai_precision': genai_metrics['precision'],
                'ml_recall': ml_metrics['recall'],
                'genai_recall': genai_metrics['recall'],
                'ml_f1': ml_metrics['f1_score'],
                'genai_f1': genai_metrics['f1_score']
            }
        }
        
        # Add latency comparison if available
        if 'inference_latency_p95' in ml_metrics and 'inference_latency_p95' in genai_metrics:
            comparison['latency_comparison'] = {
                'ml_p95_latency_ms': ml_metrics['inference_latency_p95'],
                'genai_p95_latency_ms': genai_metrics['inference_latency_p95'],
                'latency_advantage': 'ML' if ml_metrics['inference_latency_p95'] < 
                                      genai_metrics['inference_latency_p95'] else 'GenAI'
            }
        
        return comparison
    
    def _calculate_performance_score(
        self, 
        ml_metrics: Dict[str, float], 
        genai_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate performance scores for both models
        
        Args:
            ml_metrics: ML model metrics
            genai_metrics: GenAI model metrics
            
        Returns:
            Performance scores
        """
        # Weighted score: 70% accuracy, 20% F1, 10% latency (inverse)
        ml_score = (
            0.7 * ml_metrics['accuracy'] +
            0.2 * ml_metrics['f1_score'] +
            0.1 * (1.0 / (1.0 + ml_metrics.get('inference_latency_p95', 100) / 1000))
        )
        
        genai_score = (
            0.7 * genai_metrics['accuracy'] +
            0.2 * genai_metrics['f1_score'] +
            0.1 * (1.0 / (1.0 + genai_metrics.get('inference_latency_p95', 100) / 1000))
        )
        
        return {
            'ml_performance_score': ml_score,
            'genai_performance_score': genai_score,
            'performance_ratio': genai_score / ml_score if ml_score > 0 else 0
        }
    
    def _determine_winner(
        self, 
        comparison: Dict[str, Any], 
        safety_report: Dict[str, Any] = None
    ) -> str:
        """
        Determine winner based on comprehensive evaluation
        
        Args:
            comparison: Comparison metrics
            safety_report: Safety validation report
            
        Returns:
            Winner identifier
        """
        # Get performance scores
        perf_scores = comparison.get('performance_score', {})
        ml_score = perf_scores.get('ml_performance_score', 0)
        genai_score = perf_scores.get('genai_performance_score', 0)
        
        # Consider safety violations
        safety_penalty = 0.0
        if safety_report and safety_report.get('violation_rate', 0) > 0.1:
            safety_penalty = 0.2  # 20% penalty for high violation rate
        
        genai_score_adjusted = genai_score * (1 - safety_penalty)
        
        # Determine winner
        if ml_score > genai_score_adjusted * 1.1:  # ML is >10% better
            return "ML_MODEL"
        elif genai_score_adjusted > ml_score * 1.1:  # GenAI is >10% better
            return "GENAI_MODEL"
        else:
            return "TIE"
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """
        Save comparison results to disk
        
        Args:
            results: Comparison results
        """
        output_dir = Path(self.metrics_tracker.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as JSON
        filename = f"comparison_{results['comparison_id']}.json"
        output_path = output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Saved comparison results to {output_path}")
    
    def get_comparison_history(self) -> List[Dict[str, Any]]:
        """Get history of all comparisons"""
        return self.comparison_history
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from all comparisons
        
        Returns:
            Summary statistics
        """
        if not self.comparison_history:
            return {}
        
        summaries = []
        for comp in self.comparison_history:
            summary = {
                'timestamp': comp['timestamp'],
                'winner': comp['winner'],
                'ml_accuracy': comp['ml_model']['results']['metrics']['accuracy'],
                'genai_accuracy': comp['genai_model']['results']['metrics']['accuracy'],
                'accuracy_difference': comp['comparison']['accuracy_difference']
            }
            summaries.append(summary)
        
        # Calculate statistics
        ml_accuracies = [s['ml_accuracy'] for s in summaries]
        genai_accuracies = [s['genai_accuracy'] for s in summaries]
        accuracy_diffs = [s['accuracy_difference'] for s in summaries]
        
        return {
            'total_comparisons': len(summaries),
            'ml_wins': sum(1 for s in summaries if s['winner'] == 'ML_MODEL'),
            'genai_wins': sum(1 for s in summaries if s['winner'] == 'GENAI_MODEL'),
            'ties': sum(1 for s in summaries if s['winner'] == 'TIE'),
            'average_ml_accuracy': float(np.mean(ml_accuracies)),
            'average_genai_accuracy': float(np.mean(genai_accuracies)),
            'average_accuracy_difference': float(np.mean(accuracy_diffs)),
            'std_accuracy_difference': float(np.std(accuracy_diffs)),
            'ml_best_accuracy': float(np.max(ml_accuracies)),
            'genai_best_accuracy': float(np.max(genai_accuracies))
        }
