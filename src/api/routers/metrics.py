"""
Metrics router for performance monitoring
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
import json
from pathlib import Path
import logging

from src.api.dependencies import get_model_registry
from src.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_metrics(
    registry: ModelRegistry = Depends(get_model_registry)
) -> Dict[str, Any]:
    """
    Get p95 latency and last evaluation results
    
    Returns:
        Metrics including latency and evaluation results
    """
    try:
        metrics_file = Path("artifacts/evaluation_results.json")
        
        if not metrics_file.exists():
            return {
                "status": "no_metrics_available",
                "message": "No evaluation results found. Run evaluation first.",
                "model_status": registry.get_model_status()
            }
        
        # Load latest metrics
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        if not all_metrics:
            return {
                "status": "no_metrics_available",
                "message": "Evaluation results file is empty",
                "model_status": registry.get_model_status()
            }
        
        # Get latest evaluation for each model
        latest_metrics = {}
        for metric in all_metrics[-10:]:  # Last 10 evaluations
            model_name = metric['model_name']
            eval_type = metric['eval_type']
            key = f"{model_name}_{eval_type}"
            
            if key not in latest_metrics:
                latest_metrics[key] = metric
        
        # Extract p95 latency from latest metrics
        latency_metrics = {}
        for key, metric in latest_metrics.items():
            if 'metrics' in metric and 'inference_latency_p95' in metric['metrics']:
                latency_metrics[key] = {
                    'p95_latency_ms': metric['metrics']['inference_latency_p95'],
                    'timestamp': metric['timestamp']
                }
        
        return {
            "status": "success",
            "timestamp": all_metrics[-1]['timestamp'] if all_metrics else None,
            "latest_evaluations": list(latest_metrics.values()),
            "latency_metrics": latency_metrics,
            "total_evaluations": len(all_metrics),
            "model_status": registry.get_model_status()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )


@router.get("/metrics/history")
async def get_metrics_history(
    model_name: str = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get metrics history
    
    Args:
        model_name: Filter by model name
        limit: Maximum number of results
        
    Returns:
        List of historical metrics
    """
    try:
        metrics_file = Path("artifacts/evaluation_results.json")
        
        if not metrics_file.exists():
            return []
        
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        # Filter by model name if specified
        if model_name:
            filtered = [m for m in all_metrics if model_name in m['model_name']]
        else:
            filtered = all_metrics
        
        # Apply limit
        return filtered[-limit:]
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics history: {str(e)}"
        )


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary statistics of all metrics
    
    Returns:
        Summary statistics
    """
    try:
        metrics_file = Path("artifacts/evaluation_results.json")
        
        if not metrics_file.exists():
            return {
                "status": "no_data",
                "message": "No metrics data available"
            }
        
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
        
        if not all_metrics:
            return {
                "status": "empty",
                "message": "Metrics file is empty"
            }
        
        # Calculate summary statistics
        import numpy as np
        
        # Group by model
        model_metrics = {}
        for metric in all_metrics:
            model = metric['model_name']
            if model not in model_metrics:
                model_metrics[model] = []
            model_metrics[model].append(metric['metrics'])
        
        # Calculate statistics per model
        summary = {}
        for model, metrics_list in model_metrics.items():
            # Extract key metrics
            accuracies = [m.get('accuracy', 0) for m in metrics_list]
            f1_scores = [m.get('f1_score', 0) for m in metrics_list]
            latencies = [m.get('inference_latency_p95', 0) for m in metrics_list 
                        if 'inference_latency_p95' in m]
            
            summary[model] = {
                'evaluation_count': len(metrics_list),
                'accuracy': {
                    'mean': float(np.mean(accuracies)),
                    'std': float(np.std(accuracies)),
                    'min': float(np.min(accuracies)),
                    'max': float(np.max(accuracies))
                },
                'f1_score': {
                    'mean': float(np.mean(f1_scores)),
                    'std': float(np.std(f1_scores)),
                    'min': float(np.min(f1_scores)),
                    'max': float(np.max(f1_scores))
                }
            }
            
            if latencies:
                summary[model]['latency_p95'] = {
                    'mean': float(np.mean(latencies)),
                    'std': float(np.std(latencies)),
                    'min': float(np.min(latencies)),
                    'max': float(np.max(latencies))
                }
        
        return {
            "status": "success",
            "total_evaluations": len(all_metrics),
            "models_evaluated": list(model_metrics.keys()),
            "summary_by_model": summary,
            "first_evaluation": all_metrics[0]['timestamp'] if all_metrics else None,
            "last_evaluation": all_metrics[-1]['timestamp'] if all_metrics else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics summary: {str(e)}"
        )
