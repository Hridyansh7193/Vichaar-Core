"""
Configuration — all secrets loaded from environment variables.
No hardcoded keys. Works without any API key (heuristic fallback mode).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API key — supports HF_TOKEN, OPENROUTER_API_KEY, and OPENAI_API_KEY
OPENAI_API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("HF_TOKEN")
)

# API base — for LLM completions (OpenRouter / OpenAI / HF)
OPENAI_API_BASE = (
    os.getenv("OPENAI_API_BASE")
) # Removed API_BASE_URL fallback because that points to the local HTTP server, not the LLM!

# Model name
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
