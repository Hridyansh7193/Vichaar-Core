"""
Agent configuration — definitions, hyperparameters, phase strategy.
"""
from typing import Dict, Any

# -- Agent Definitions ---------------------------------------------------
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

# -- Learning Hyperparameters --------------------------------------------
ALPHA = 0.3
BASE_EPSILON = 0.15
CRISIS_EPSILON = 0.4
SOFTMAX_TEMP = 0.5
REPEAT_PENALTY = 0.3
AGREEMENT_BONUS = 0.2
DIVERSITY_BONUS = 0.1
MEMORY_AVOIDANCE = 0.15
MEMORY_CAPACITY = 200

# -- Phase Strategy Modifiers --------------------------------------------
PHASE_STRATEGY = {
    "morning":   {"epsilon_mult": 1.4, "label": "EXPLORE",  "diversity_mult": 2.0, "repeat_mult": 0.5},
    "execution": {"epsilon_mult": 0.5, "label": "OPTIMIZE", "diversity_mult": 0.5, "repeat_mult": 1.0},
    "review":    {"epsilon_mult": 0.8, "label": "CORRECT",  "diversity_mult": 1.0, "repeat_mult": 1.5},
    "planning":  {"epsilon_mult": 1.0, "label": "PLAN",     "diversity_mult": 1.5, "repeat_mult": 0.8},
}

# -- Metric Display Names -----------------------------------------------
METRIC_LABELS = {
    "expected_profit": "profit",
    "legal_risk": "legal risk",
    "env_impact": "env impact",
    "public_sentiment": "sentiment",
    "cost": "cost",
}
