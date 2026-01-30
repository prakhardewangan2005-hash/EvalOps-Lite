"""
Serialization utilities for models and data
"""

import pickle
import json
import numpy as np
from typing import Any, Dict, List
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_model(model: Any, filepath: str) -> None:
    """
    Save model to disk using joblib
    
    Args:
        model: Model to save
        filepath: Path to save model
    """
    try:
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise


def load_model(filepath: str) -> Any:
    """
    Load model from disk
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """
    Save data as JSON with numpy support
    
    Args:
        data: Data to save
        filepath: Path to save file
        indent: JSON indentation
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=indent)
        logger.debug(f"JSON saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {str(e)}")
        raise


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.debug(f"JSON loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON: {str(e)}")
        raise


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data using pickle
    
    Args:
        data: Data to save
        filepath: Path to save file
    """
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.debug(f"Pickle saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save pickle: {str(e)}")
        raise


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded data
    """
    try:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Pickle file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.debug(f"Pickle loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Failed to load pickle: {str(e)}")
        raise
