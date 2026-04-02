"""
Centralized configuration for the Vichaar-Core RL environment.
All tunables live here so experiments can override them cleanly.
"""
from typing import Dict, Any, List

# ── Action Space ──────────────────────────────────────────────────────
ACTIONS: List[str] = [
    "increase_production",
    "delay_launch",
    "invest_in_safety",
    "launch_fast",
    "reduce_cost",
    "lobby_regulators",
    "pr_campaign",
    "green_innovation",
    "outsource_tasks",
    "employee_training",
    "market_research",
    "vulnerability_audit",
]

# ── Agent Definitions ─────────────────────────────────────────────────
AGENT_DEFS: Dict[str, Dict[str, Any]] = {
    "profit": {
        "desc": "Maximize profit aggressively. Risk-tolerant, focus on revenue and cost efficiency.",
        "reward_weights": {"expected_profit": 2.0, "cost": -1.0},
        "preferred_actions": ["increase_production", "launch_fast", "reduce_cost", "market_research"],
    },
    "ethics": {
        "desc": "Minimize environmental impact and social harm. Advocate for sustainability.",
        "reward_weights": {"env_impact": -1.5, "legal_risk": -0.5},
        "preferred_actions": ["green_innovation", "invest_in_safety", "employee_training"],
    },
    "pr": {
        "desc": "Maximize public sentiment and protect brand image at all costs.",
        "reward_weights": {"public_sentiment": 2.0},
        "preferred_actions": ["pr_campaign", "delay_launch", "employee_training", "green_innovation"],
    },
    "legal": {
        "desc": "Minimize legal risk and ensure full regulatory compliance.",
        "reward_weights": {"legal_risk": -2.0},
        "preferred_actions": ["vulnerability_audit", "invest_in_safety", "lobby_regulators"],
    },
    "risk": {
        "desc": "Balance all metrics to ensure long-term system resilience and stability.",
        "reward_weights": {"legal_risk": -1.0, "env_impact": -1.0, "cost": -1.0},
        "preferred_actions": ["invest_in_safety", "delay_launch", "vulnerability_audit", "reduce_cost"],
    },
}

# ── Action Effects on Metrics ─────────────────────────────────────────
# Each action maps to {metric_name: delta_value}
ACTION_EFFECTS: Dict[str, Dict[str, float]] = {
    "increase_production":  {"expected_profit": 0.08, "env_impact": 0.05, "cost": 0.03},
    "delay_launch":         {"public_sentiment": 0.10, "expected_profit": -0.05, "cost": 0.02},
    "invest_in_safety":     {"legal_risk": -0.10, "env_impact": -0.05, "cost": 0.06},
    "launch_fast":          {"expected_profit": 0.15, "legal_risk": 0.10, "public_sentiment": -0.10},
    "reduce_cost":          {"cost": -0.10, "expected_profit": 0.02, "public_sentiment": -0.05},
    "lobby_regulators":     {"legal_risk": -0.05, "cost": 0.03, "public_sentiment": -0.02},
    "pr_campaign":          {"public_sentiment": 0.12, "cost": 0.04},
    "green_innovation":     {"env_impact": -0.12, "expected_profit": 0.03, "cost": 0.08},
    "outsource_tasks":      {"cost": -0.05, "legal_risk": 0.03},
    "employee_training":    {"public_sentiment": 0.05, "expected_profit": 0.02, "cost": 0.03},
    "market_research":      {"expected_profit": 0.04, "cost": 0.02},
    "vulnerability_audit":  {"legal_risk": -0.08, "cost": 0.02},
}

# ── Event Definitions ─────────────────────────────────────────────────
EVENT_DEFS: Dict[str, Dict[str, Any]] = {
    "regulatory_crisis": {
        "prob": 0.08,
        "effects": {"legal_risk": 0.15, "public_sentiment": -0.10},
    },
    "market_opportunity": {
        "prob": 0.10,
        "effects": {"expected_profit": 0.10},
    },
    "competitor_move": {
        "prob": 0.06,
        "effects": {"expected_profit": -0.08, "cost": 0.05},
    },
    "media_scandal": {
        "prob": 0.04,
        "effects": {"public_sentiment": -0.20, "legal_risk": 0.05},
    },
    "supply_disruption": {
        "prob": 0.05,
        "effects": {"cost": 0.10, "expected_profit": -0.05},
    },
}

# ── Phase Cycle ───────────────────────────────────────────────────────
PHASES = ["morning", "execution", "review", "planning"]

# ── Grading Weights ───────────────────────────────────────────────────
GRADE_WEIGHTS = {
    "expected_profit": 0.4,
    "legal_risk": -0.3,
    "env_impact": -0.2,
    "public_sentiment": 0.1,
}

# ── Default Env Settings ──────────────────────────────────────────────
DEFAULT_MAX_STEPS = 15
DEFAULT_SEED = 42
MEMORY_CAPACITY = 200
COLLABORATION_BONUS = 0.01  # bonus per step when agents collaborate
