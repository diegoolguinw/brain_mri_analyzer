# ── Build stage: install CPU-only PyTorch (keeps image small) ────────
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for Pillow / matplotlib / reportlab
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc libjpeg62-turbo-dev zlib1g-dev libfreetype6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps (CPU-only torch to save ~1.5 GB)
COPY brain_app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY brain_app/ /app/

# Copy model checkpoint into the image
COPY checkpoints/ /app/checkpoints/

# Collect static files
RUN python manage.py collectstatic --noinput

# Create media directories
RUN mkdir -p /app/media/uploads /app/media/results

EXPOSE 8000

CMD ["gunicorn", "brain_app.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
