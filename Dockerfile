FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (.dockerignore excludes secrets/caches)
COPY . .

# Create trajectory output directory
RUN mkdir -p trajectories training_trajectories

# Environment variables — set at runtime via docker run -e or HF Spaces secrets
# No defaults for secrets; the system works in heuristic mode without any API key.
ENV HF_TOKEN=""
ENV API_BASE_URL=""
ENV MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"

# HF Spaces uses port 7860; OpenEnv validator also checks 7860
EXPOSE 7860

# Entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
