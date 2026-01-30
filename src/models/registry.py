"""
Model registry for managing ML and GenAI models
"""

import joblib
from typing import Dict, Any, Optional
import logging
from pathlib import Path

from src.models.ml_model import MLModel
from src.models.genai_baseline import GenAIBaseline, GenAIConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing and serving models
    """
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        """
        Initialize model registry
        
        Args:
            artifacts_dir: Directory for model artifacts
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(exist_ok=True)
        
        self.ml_model: Optional[MLModel] = None
        self.genai_model: Optional[GenAIBaseline] = None
        self.preprocessor = None
        
    def load_ml_model(self, model_path: Optional[str] = None) -> MLModel:
        """
        Load ML model from disk
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded ML model
        """
        if model_path is None:
            model_path = self.artifacts_dir / "model.pkl"
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.ml_model = MLModel(model_type='random_forest')
        self.ml_model.load(model_path)
        
        logger.info(f"Loaded ML model from {model_path}")
        return self.ml_model
    
    def load_preprocessor(self, preprocessor_path: Optional[str] = None) -> Any:
        """
        Load preprocessor from disk
        
        Args:
            preprocessor_path: Path to preprocessor file
            
        Returns:
            Loaded preprocessor
        """
        if preprocessor_path is None:
            preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        
        if not Path(preprocessor_path).exists():
            raise FileNotFoundError(f"Preprocessor file not found: {preprocessor_path}")
        
        self.preprocessor = joblib.load(preprocessor_path)
        
        logger.info(f"Loaded preprocessor from {preprocessor_path}")
        return self.preprocessor
    
    def initialize_genai_model(self, config: Optional[GenAIConfig] = None) -> GenAIBaseline:
        """
        Initialize GenAI model
        
        Args:
            config: GenAI configuration
            
        Returns:
            Initialized GenAI model
        """
        self.genai_model = GenAIBaseline(config)
        self.genai_model.load_model()
        
        logger.info("Initialized GenAI model")
        return self.genai_model
    
    def save_model(self, ml_model: MLModel, preprocessor: Any, model_name: str = "model.pkl"):
        """
        Save model and preprocessor to disk
        
        Args:
            ml_model: Trained ML model
            preprocessor: Fitted preprocessor
            model_name: Name for model file
        """
        # Save ML model
        model_path = self.artifacts_dir / model_name
        ml_model.save(str(model_path))
        
        # Save preprocessor
        preprocessor_path = self.artifacts_dir / "preprocessor.pkl"
        joblib.dump(preprocessor, preprocessor_path)
        
        logger.info(f"Saved model artifacts to {self.artifacts_dir}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status of all models in registry
        
        Returns:
            Dictionary with model status
        """
        status = {
            'ml_model': {
                'loaded': self.ml_model is not None,
                'type': self.ml_model.model_type if self.ml_model else None,
                'trained': self.ml_model.is_trained if self.ml_model else False
            },
            'genai_model': {
                'loaded': self.genai_model is not None,
                'config': self.genai_model.get_config() if self.genai_model else None
            },
            'preprocessor': {
                'loaded': self.preprocessor is not None
            },
            'artifacts_dir': str(self.artifacts_dir)
        }
        
        return status
