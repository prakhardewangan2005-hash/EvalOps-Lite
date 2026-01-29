"""
EvalOps Lite - Production-grade ML + GenAI evaluation framework
"""

__version__ = "1.0.0"
__author__ = "Microsoft MLE Intern Candidate"

from src.data.ingestion import DataIngestor
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.comparator import ModelComparator
from src.api.app import create_app

__all__ = [
    "DataIngestor",
    "PreprocessingPipeline",
    "MLModel",
    "GenAIBaseline",
    "ModelEvaluator",
    "ModelComparator",
    "create_app",
]
