# ── Build stage: lightweight ONNX Runtime deployment ─────────────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for Pillow / matplotlib / reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libjpeg62-turbo-dev zlib1g-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (no PyTorch — using onnxruntime instead, saves ~250 MB)
COPY brain_app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY brain_app/ /app/

# Copy ONNX model + metadata (NOT the .pt checkpoint)
COPY checkpoints/model.onnx checkpoints/model_meta.json /app/checkpoints/

# Collect static files
RUN python manage.py collectstatic --noinput

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/results

EXPOSE 8000

# Single worker to minimize memory; preload to share model across requests
CMD ["gunicorn", "brain_app.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "120"]
