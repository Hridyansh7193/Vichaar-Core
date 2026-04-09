"""
Configuration — all secrets loaded from environment variables.
No hardcoded keys. Works without any API key (heuristic fallback mode).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API key — supports API_KEY (from OpenEnv), HF_TOKEN, OPENROUTER_API_KEY, and OPENAI_API_KEY
OPENAI_API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)

# API base — for LLM completions (OpenRouter / OpenAI / HF / OpenEnv LiteLLM proxy)
OPENAI_API_BASE = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
)

# Model name
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
