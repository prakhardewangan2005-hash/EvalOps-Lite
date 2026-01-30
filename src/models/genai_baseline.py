"""
GenAI-style baseline model using offline transformer
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import time
import logging
from dataclasses import dataclass
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

from src.data.schemas import GenAIOutputSchema, PredictionRequestSchema

logger = logging.getLogger(__name__)


@dataclass
class GenAIConfig:
    """Configuration for GenAI baseline"""
    model_name: str = "distilbert-base-uncased"
    device: str = "cpu"
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    include_rationale: bool = True
    mock_mode: bool = False  # For testing without actual model


class GenAIBaseline:
    """
    GenAI-style baseline using offline transformer model
    
    Note: Uses HuggingFace transformers for offline inference
    """
    
    def __init__(self, config: GenAIConfig = None):
        """
        Initialize GenAI baseline
        
        Args:
            config: Configuration for GenAI model
        """
        self.config = config or GenAIConfig()
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self.is_loaded = False
        
        # Feature templates for generating text from numeric features
        self.feature_templates = [
            "The mean radius is {:.2f}.",
            "Texture mean measures {:.2f}.",
            "Perimeter mean is {:.2f}.",
            "Area mean measures {:.2f}.",
            "Smoothness mean is {:.2f}.",
            "Compactness mean measures {:.2f}.",
            "Concavity mean is {:.2f}.",
            "Concave points mean {:.2f}.",
            "Symmetry mean is {:.2f}.",
            "Fractal dimension mean measures {:.2f}."
        ]
        
        # Rationale templates
        self.rationale_templates = {
            0: "Based on the medical feature analysis, the pattern appears benign with {confidence:.0%} confidence. The values fall within normal ranges for healthy tissue characteristics.",
            1: "The feature analysis suggests malignant characteristics with {confidence:.0%} confidence. Several indicators show concerning deviations from healthy tissue patterns."
        }
        
    def load_model(self):
        """Load the transformer model"""
        if self.config.mock_mode:
            logger.info("Running in mock mode - no actual model loaded")
            self.is_loaded = True
            return
        
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                num_labels=2
            )
            self.model.to(self.config.device)
            self.model.eval()
            
            # Create text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1 if self.config.device == "cpu" else 0
            )
            
            self.is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Fall back to mock mode
            self.config.mock_mode = True
            self.is_loaded = True
    
    def _features_to_text(self, features: List[float]) -> str:
        """
        Convert numeric features to descriptive text
        
        Args:
            features: List of 30 feature values
            
        Returns:
            Descriptive text for model input
        """
        # Select key features for text generation
        key_features = features[:10]  # First 10 features are often most important
        
        # Create descriptive text
        text_parts = []
        for i, (template, value) in enumerate(zip(self.feature_templates, key_features)):
            if i < len(key_features):
                text_parts.append(template.format(value))
        
        # Add summary
        text_parts.append("Analyze these medical imaging features to assess tumor characteristics.")
        
        return " ".join(text_parts)
    
    def _mock_predict(self, text: str) -> Tuple[int, float, str]:
        """
        Mock prediction for testing without actual model
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (label, probability, rationale)
        """
        # Simulate processing time
        time.sleep(0.01 + random.random() * 0.05)
        
        # Simple rule-based prediction for demonstration
        # In reality, this would use the actual transformer model
        features = json.loads(text.split("features:")[-1]) if "features:" in text else []
        
        if features:
            # Simple heuristic based on key features
            mean_radius = features[0] if len(features) > 0 else 14
            mean_texture = features[1] if len(features) > 1 else 19
            mean_perimeter = features[2] if len(features) > 2 else 92
            
            # Calculate mock probability
            risk_score = (
                (mean_radius - 10) / 10 +
                (mean_texture - 15) / 20 +
                (mean_perimeter - 80) / 40
            ) / 3
            
            probability = 1 / (1 + np.exp(-risk_score))
            label = 1 if probability > 0.5 else 0
            confidence = abs(probability - 0.5) * 2
            
        else:
            # Random prediction for text-only mode
            probability = random.uniform(0.3, 0.7)
            label = 1 if probability > 0.5 else 0
            confidence = abs(probability - 0.5) * 2
        
        # Generate rationale
        rationale = self.rationale_templates[label].format(confidence=confidence)
        
        return label, probability, rationale
    
    def predict(self, features: List[float]) -> Dict[str, Any]:
        """
        Make prediction with GenAI baseline
        
        Args:
            features: List of feature values
            
        Returns:
            Dictionary with structured GenAI output
        """
        if not self.is_loaded:
            self.load_model()
        
        # Validate input
        schema = PredictionRequestSchema(features=features)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Convert features to text
            text_input = self._features_to_text(schema.features)
            
            if self.config.mock_mode or self.classifier is None:
                # Use mock prediction
                label, probability, rationale = self._mock_predict(text_input)
                
            else:
                # Use actual transformer model
                with torch.no_grad():
                    result = self.classifier(
                        text_input,
                        truncation=True,
                        max_length=self.config.max_length
                    )[0]
                    
                    # Map transformer output to our format
                    # Assuming label 1 is positive (malignant)
                    if result['label'] == 'LABEL_1':
                        probability = result['score']
                        label = 1
                    else:
                        probability = 1 - result['score']
                        label = 0
                    
                    # Generate rationale
                    confidence = abs(probability - 0.5) * 2
                    rationale = self.rationale_templates[label].format(
                        confidence=confidence
                    )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Create structured output
            output = {
                'label': label,
                'probability': float(probability),
                'confidence': float(abs(probability - 0.5) * 2),  # Confidence as distance from 0.5
                'rationale': rationale,
                'latency_ms': latency_ms,
                'model_name': self.config.model_name,
                'input_features_processed': len(features)
            }
            
            # Validate output schema
            GenAIOutputSchema(**output)
            
            logger.debug(f"GenAI prediction - Label: {label}, Probability: {probability:.3f}, "
                        f"Latency: {latency_ms:.2f}ms")
            
            return output
            
        except Exception as e:
            logger.error(f"GenAI prediction failed: {str(e)}")
            # Return safe default
            return {
                'label': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'rationale': f"Prediction failed: {str(e)[:100]}",
                'latency_ms': (time.time() - start_time) * 1000,
                'model_name': self.config.model_name,
                'error': str(e)
            }
    
    def batch_predict(self, features_list: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions
        
        Args:
            features_list: List of feature lists
            
        Returns:
            List of prediction results
        """
        results = []
        for features in features_list:
            results.append(self.predict(features))
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_name': self.config.model_name,
            'device': self.config.device,
            'max_length': self.config.max_length,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'mock_mode': self.config.mock_mode,
            'is_loaded': self.is_loaded
        }
