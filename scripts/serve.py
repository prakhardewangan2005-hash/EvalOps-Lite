#!/usr/bin/env python3
"""
API server script for EvalOps Lite
"""

import sys
import os
import logging
from pathlib import Path
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.app import create_app
from src.utils.logging import setup_logging
from configs.settings import settings

# Setup logging
setup_logging(log_level=settings.log_level)
logger = logging.getLogger(__name__)


def serve():
    """Start the FastAPI server"""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Host: {settings.api_host}, Port: {settings.api_port}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Create app
    app = create_app()
    
    # Start server
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    serve()
