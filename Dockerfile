# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (optional, but keep update)
RUN apt-get update

# Copy all project files (including requirements)
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000 11000 8001

# Install dependencies and start app at runtime
CMD echo "=== Installing dependencies ===" && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "=== Running app ===" && \
    python3 manage.py
