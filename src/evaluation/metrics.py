"""
Metrics calculation and tracking utilities
"""

import numpy as np
from typing import Dict, Any, List
import json
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Tracks and persists evaluation metrics
    """
    
    def __init__(self, output_dir: str = "artifacts"):
        """
        Initialize metrics tracker
        
        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics_history: List[Dict[str, Any]] = []
        
    def add_metrics(self, metrics: Dict[str, Any], model_name: str, eval_type: str):
        """
        Add metrics to history
        
        Args:
            metrics: Metrics dictionary
            model_name: Name of the model
            eval_type: Type of evaluation
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'eval_type': eval_type,
            'metrics': metrics
        }
        
        self.metrics_history.append(entry)
        logger.info(f"Added metrics for {model_name} ({eval_type})")
    
    def save_metrics(self, filename: str = "evaluation_results.json"):
        """
        Save metrics to JSON file
        
        Args:
            filename: Name of output file
        """
        output_path = self.output_dir / filename
        
        # Prepare data for JSON serialization
        serializable_data = []
        for entry in self.metrics_history:
            serializable_entry = {
                'timestamp': entry['timestamp'],
                'model_name': entry['model_name'],
                'eval_type': entry['eval_type'],
                'metrics': self._make_serializable(entry['metrics'])
            }
            serializable_data.append(serializable_entry)
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        logger.info(f"Saved metrics to {output_path}")
    
    def load_metrics(self, filename: str = "evaluation_results.json"):
        """
        Load metrics from JSON file
        
        Args:
            filename: Name of input file
        """
        input_path = self.output_dir / filename
        
        if not input_path.exists():
            logger.warning(f"Metrics file not found: {input_path}")
            return
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        self.metrics_history = data
        logger.info(f"Loaded {len(data)} metrics entries from {input_path}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_latest_metrics(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get latest metrics for a model
        
        Args:
            model_name: Name of model to filter by
            
        Returns:
            Latest metrics dictionary
        """
        if not self.metrics_history:
            return {}
        
        filtered = self.metrics_history
        if model_name:
            filtered = [entry for entry in self.metrics_history 
                       if entry['model_name'] == model_name]
        
        if not filtered:
            return {}
        
        return filtered[-1]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of all metrics
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {}
        
        summary = {
            'total_evaluations': len(self.metrics_history),
            'models_evaluated': list(set(entry['model_name'] for entry in self.metrics_history)),
            'latest_evaluation': self.metrics_history[-1]['timestamp'],
            'average_metrics': {}
        }
        
        # Calculate average metrics per model
        model_metrics = {}
        for entry in self.metrics_history:
            model_name = entry['model_name']
            if model_name not in model_metrics:
                model_metrics[model_name] = []
            model_metrics[model_name].append(entry['metrics'])
        
        # Calculate averages
        for model_name, metrics_list in model_metrics.items():
            avg_metrics = {}
            for metric_name in metrics_list[0].keys():
                values = [m.get(metric_name) for m in metrics_list if metric_name in m]
                if values:
                    # Filter out None values
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        avg_metrics[metric_name] = float(np.mean(valid_values))
            
            summary['average_metrics'][model_name] = avg_metrics
        
        return summary
