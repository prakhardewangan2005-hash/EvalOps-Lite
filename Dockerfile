# ================================
# Production-ready Dockerfile
# Python 3.11 | FastAPI | Railway
# ================================

FROM python:3.11-slim

# Prevent Python from writing pyc files & enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal & safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy dependency files first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Railway uses 8000 by default)
EXPOSE 8000

# Start FastAPI app (adjust module if needed)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "scripts.serve:app", "--bind", "0.0.0.0:8000"]
