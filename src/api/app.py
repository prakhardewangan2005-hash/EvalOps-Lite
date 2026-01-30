"""
FastAPI application for model serving
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from src.api.routers import predict, metrics
from src.api.dependencies import get_model_registry
from src.models.registry import ModelRegistry
from src.utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global model registry
model_registry: ModelRegistry = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    
    Handles startup and shutdown events
    """
    global model_registry
    
    # Startup
    logger.info("Starting EvalOps Lite API server...")
    
    try:
        # Initialize model registry
        model_registry = ModelRegistry(artifacts_dir="artifacts")
        
        # Load ML model
        model_registry.load_ml_model()
        
        # Load preprocessor
        model_registry.load_preprocessor()
        
        # Initialize GenAI model
        from src.models.genai_baseline import GenAIConfig
        genai_config = GenAIConfig(mock_mode=True)  # Use mock for demo
        model_registry.initialize_genai_model(genai_config)
        
        logger.info("Model registry initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize model registry: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down EvalOps Lite API server...")
    model_registry = None


# Create FastAPI app
app = FastAPI(
    title="EvalOps Lite API",
    description="Production-grade ML + GenAI evaluation and deployment service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path
        }
    )

# Include routers
app.include_router(
    predict.router,
    prefix="/api/v1",
    tags=["predictions"]
)

app.include_router(
    metrics.router,
    prefix="/api/v1",
    tags=["metrics"]
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "EvalOps Lite",
        "version": "1.0.0",
        "description": "Production ML + GenAI evaluation system",
        "endpoints": {
            "predict": "/api/v1/predict",
            "metrics": "/api/v1/metrics",
            "docs": "/docs",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model_registry
    
    status = {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z"  # Would use datetime in production
    }
    
    if model_registry:
        status.update({
            "model_registry": model_registry.get_model_status(),
            "service": "EvalOps Lite API",
            "version": "1.0.0"
        })
    
    return status


@app.get("/api/v1/models/status")
async def model_status():
    """Get status of all loaded models"""
    global model_registry
    
    if not model_registry:
        raise HTTPException(
            status_code=503,
            detail="Model registry not initialized"
        )
    
    return {
        "status": "available",
        "models": model_registry.get_model_status()
    }
