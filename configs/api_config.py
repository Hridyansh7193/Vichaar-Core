"""
Configuration — all secrets loaded from environment variables.
No hardcoded keys. Works without any API key (heuristic fallback mode).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API key — supports HF_TOKEN, OPENROUTER_API_KEY, and OPENAI_API_KEY
OPENAI_API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

# API base — supports API_BASE_URL (OpenEnv standard) and OPENAI_API_BASE
OPENAI_API_BASE = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or "https://api-inference.huggingface.co/v1"
)

# Model name
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
