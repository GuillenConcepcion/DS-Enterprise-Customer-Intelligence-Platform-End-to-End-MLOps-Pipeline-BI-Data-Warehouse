# Use official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies (including git for mlflow/shap if needed, and curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy configuration files to cache docker layer
COPY pyproject.toml uv.lock /app/

# Install all dependencies using uv (respecting uv.lock)
RUN uv pip install --system --no-cache -r pyproject.toml

# Create a non-root user for security
RUN useradd -u 10001 --create-home mlops_user && \
    chown -R mlops_user:mlops_user /app

# Copy the entire project into the container
COPY --chown=mlops_user:mlops_user . /app/

# Switch to the non-root user
USER mlops_user

# Expose port 8000 for FastAPI
EXPOSE 8000

# Healthcheck to monitor API status
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Run FastAPI using Uvicorn
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

