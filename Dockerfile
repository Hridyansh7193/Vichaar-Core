FROM python:3.12-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Create runtime directories
RUN mkdir -p trajectories training_trajectories

# Environment variables (overridden at runtime via HF Spaces secrets)
ENV HF_TOKEN=""
ENV OPENAI_API_KEY=""
ENV OPENROUTER_API_KEY=""
ENV API_BASE_URL=""
ENV MODEL_NAME=""

EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
