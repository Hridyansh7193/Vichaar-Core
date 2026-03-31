import os
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# API CONFIGURATION
# ==========================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", os.getenv("API_BASE_URL", "https://api.openai.com/v1"))
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# ==========================================
# ENVIRONMENT DEFAULTS
# ==========================================
DEFAULT_METRICS = {
    "cost": 50000.0,
    "expected_profit": 1500000.0,
    "legal_risk": 0.85,
    "env_impact": 0.05,
    "public_sentiment": -0.6
}

DEFAULT_UNCERTAINTY = {
    "legal_conf": 0.5,
    "env_conf": 0.95
}

# ==========================================
# GRADER / REWARD SYSTEM (NORMALIZED)
# ==========================================
# Upper bound cap for profit normalization
PROFIT_CAP = 1000000.0

# Base Weights (Summing to 1.0 or similar)
PROFIT_WEIGHT = 0.4
LEGAL_RISK_WEIGHT = 0.3
ENV_IMPACT_WEIGHT = 0.2
SENTIMENT_WEIGHT = 0.1

# Decision Adjustments
RISK_THRESHOLD = 0.7
LAUNCH_PENALTY_VALUE = 0.5
MODIFY_BONUS_VALUE = 0.2
REJECT_REWARD_VALUE = 0.3
