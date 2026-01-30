#!/usr/bin/env python3
"""
Evaluation script for EvalOps Lite
Compares ML and GenAI models comprehensively
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.registry import ModelRegistry
from src.models.genai_baseline import GenAIConfig
from src.evaluation.comparator import ModelComparator
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def evaluate_models():
    """Main evaluation function"""
    logger.info("Starting EvalOps Lite evaluation pipeline...")
    
    try:
        # 1. Load data
        logger.info("Step 1: Loading data...")
        ingestor = DataIngestor(test_size=0.2, random_state=42)
        dataset = ingestor.load_breast_cancer_data()
        
        # 2. Load models
        logger.info("Step 2: Loading models...")
        registry = ModelRegistry(artifacts_dir="artifacts")
        
        # Load ML model and preprocessor
        ml_model = registry.load_ml_model()
        preprocessor = registry.load_preprocessor()
        
        # Initialize GenAI model
        genai_config = GenAIConfig(mock_mode=True)
        genai_model = registry.initialize_genai_model(genai_config)
        
        logger.info(f"ML Model: {ml_model.model_type}")
        logger.info(f"GenAI Model: {genai_model.config.model_name}")
        logger.info(f"Preprocessor: Loaded")
        
        # 3. Preprocess test data for ML model
        logger.info("Step 3: Preprocessing test data...")
        X_test_processed = preprocessor.transform(dataset.X_test)
        
        # 4. Compare models
        logger.info("Step 4: Comparing models...")
        comparator = ModelComparator(output_dir="artifacts")
        
        results = comparator.compare_models(
            ml_model=ml_model,
            genai_model=genai_model,
            X_test=dataset.X_test,  # Raw features for GenAI
            y_test=dataset.y_test,
            run_safety_checks=True,
            save_results=True
        )
        
        # 5. Print results
        logger.info("Evaluation complete!")
        
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # ML Model results
        ml_metrics = results['ml_model']['results']['metrics']
        print(f"\nML Model ({ml_model.model_type}):")
        print(f"  Accuracy:  {ml_metrics['accuracy']:.4f}")
        print(f"  Precision: {ml_metrics['precision']:.4f}")
        print(f"  Recall:    {ml_metrics['recall']:.4f}")
        print(f"  F1 Score:  {ml_metrics['f1_score']:.4f}")
        if 'inference_latency_p95' in ml_metrics:
            print(f"  P95 Latency: {ml_metrics['inference_latency_p95']:.2f}ms")
        
        # GenAI Model results
        genai_metrics = results['genai_model']['results']['metrics']
        print(f"\nGenAI Model ({genai_model.config.model_name}):")
        print(f"  Accuracy:  {genai_metrics['accuracy']:.4f}")
        print(f"  Precision: {genai_metrics['precision']:.4f}")
        print(f"  Recall:    {genai_metrics['recall']:.4f}")
        print(f"  F1 Score:  {genai_metrics['f1_score']:.4f}")
        if 'inference_latency_p95' in genai_metrics:
            print(f"  P95 Latency: {genai_metrics['inference_latency_p95']:.2f}ms")
        
        # Safety report
        if results.get('safety_report'):
            safety = results['safety_report']
            print(f"\nSafety Report:")
            print(f"  Violation Rate: {safety['violation_rate']:.2%}")
            print(f"  Average Score:  {safety['average_score']:.2f}")
            if safety['common_violations']:
                print(f"  Common Violations:")
                for violation, count in safety['common_violations'].items():
                    print(f"    - {violation}: {count}")
        
        # Comparison
        comparison = results['comparison']
        print(f"\nComparison:")
        print(f"  Accuracy Difference: {comparison['accuracy_difference']:+.4f} "
              f"(GenAI - ML)")
        print(f"  F1 Score Difference: {comparison['f1_difference']:+.4f}")
        print(f"  Latency Ratio: {comparison['latency_ratio']:.2f}x")
        
        perf_scores = comparison['performance_score']
        print(f"  Performance Scores:")
        print(f"    ML:    {perf_scores['ml_performance_score']:.3f}")
        print(f"    GenAI: {perf_scores['genai_performance_score']:.3f}")
        
        # Winner
        print(f"\nWinner: {results['winner']}")
        
        print("="*60)
        print(f"\nResults saved to: artifacts/comparison_{results['comparison_id']}.json")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = evaluate_models()
    sys.exit(0 if success else 1)
