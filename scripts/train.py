#!/usr/bin/env python3
"""
Training script for EvalOps Lite
Trains both ML and GenAI baselines
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import DataIngestor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline, GenAIConfig
from src.models.registry import ModelRegistry
from src.evaluation.evaluator import ModelEvaluator
from src.utils.logging import setup_logging

# Setup logging
setup_logging(log_level="INFO")
logger = logging.getLogger(__name__)


def train_model():
    """Main training function"""
    logger.info("Starting EvalOps Lite training pipeline...")
    
    try:
        # 1. Load data
        logger.info("Step 1: Loading data...")
        ingestor = DataIngestor(test_size=0.2, random_state=42)
        dataset = ingestor.load_breast_cancer_data()
        
        # Print dataset statistics
        stats = ingestor.get_feature_statistics()
        logger.info(f"Dataset loaded: {stats['training_samples']} training, "
                   f"{stats['test_samples']} test samples")
        
        # 2. Preprocessing
        logger.info("Step 2: Preprocessing data...")
        preprocessor = PreprocessingPipeline()
        X_train_processed = preprocessor.fit_transform(dataset.X_train)
        X_test_processed = preprocessor.transform(dataset.X_test)
        
        logger.info(f"Preprocessing complete. Selected {X_train_processed.shape[1]} features")
        
        # 3. Train ML model
        logger.info("Step 3: Training ML model...")
        ml_model = MLModel(
            model_type='random_forest',
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        ml_model.train(X_train_processed, dataset.y_train)
        
        logger.info(f"ML model trained in {ml_model.training_time:.2f} seconds")
        
        # 4. Initialize GenAI model
        logger.info("Step 4: Initializing GenAI baseline...")
        genai_config = GenAIConfig(mock_mode=True)  # Use mock for demo
        genai_model = GenAIBaseline(genai_config)
        genai_model.load_model()
        
        logger.info(f"GenAI model initialized: {genai_config.model_name}")
        
        # 5. Evaluate models
        logger.info("Step 5: Evaluating models...")
        evaluator = ModelEvaluator()
        
        # Evaluate ML model
        ml_results = evaluator.evaluate_ml_model(
            ml_model, X_test_processed, dataset.y_test,
            calculate_latency=True
        )
        
        # Evaluate GenAI model
        genai_results = evaluator.evaluate_genai_model(
            genai_model, dataset.X_test, dataset.y_test,
            calculate_latency=True
        )
        
        logger.info("Model evaluation complete:")
        logger.info(f"ML Model - Accuracy: {ml_results['metrics']['accuracy']:.4f}, "
                   f"F1: {ml_results['metrics']['f1_score']:.4f}")
        logger.info(f"GenAI Model - Accuracy: {genai_results['metrics']['accuracy']:.4f}, "
                   f"F1: {genai_results['metrics']['f1_score']:.4f}")
        
        # 6. Save models
        logger.info("Step 6: Saving models and artifacts...")
        registry = ModelRegistry(artifacts_dir="artifacts")
        registry.save_model(ml_model, preprocessor)
        
        logger.info("Training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Dataset: Breast Cancer (n={dataset.X_train.shape[0] + dataset.X_test.shape[0]})")
        print(f"ML Model: Random Forest (accuracy={ml_results['metrics']['accuracy']:.4f})")
        print(f"GenAI Model: {genai_config.model_name} (accuracy={genai_results['metrics']['accuracy']:.4f})")
        print(f"Artifacts saved to: artifacts/")
        print("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    success = train_model()
    sys.exit(0 if success else 1)
