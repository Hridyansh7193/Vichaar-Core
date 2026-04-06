"""
Centralized environment configuration for Vichaar-Core.
All environment tunables live here.
"""
from typing import Dict, Any, List

# -- Action Space -------------------------------------------------------
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

# -- Action Effects on Metrics -------------------------------------------
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

# -- Event Definitions ---------------------------------------------------
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

# -- Phase Cycle ---------------------------------------------------------
PHASES = ["morning", "execution", "review", "planning"]

# -- Grading Weights -----------------------------------------------------
GRADE_WEIGHTS = {
    "expected_profit": 0.4,
    "legal_risk": -0.3,
    "env_impact": -0.2,
    "public_sentiment": 0.1,
}

# -- Global Score (Coordinator lookahead planning) -----------------------
GLOBAL_SCORE_WEIGHTS: Dict[str, float] = {
    "expected_profit":   1.0,
    "legal_risk":       -2.0,
    "cost":             -1.8,
    "env_impact":       -1.5,
    "public_sentiment":  0.8,
}

# -- CEO Hard Constraint Thresholds --------------------------------------
CEO_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "cost":        {"threshold": 0.80, "force_action": "reduce_cost",       "label": "COST CRISIS"},
    "legal_risk":  {"threshold": 0.65, "force_action": "invest_in_safety",  "label": "LEGAL CRISIS"},
    "env_impact":  {"threshold": 0.65, "force_action": "green_innovation",  "label": "ENV CRISIS"},
}

# -- Safe Mode -----------------------------------------------------------
SAFE_MODE_SCORE_DECLINE = 0.15
UNSAFE_ACTIONS: List[str] = ["launch_fast", "outsource_tasks", "increase_production"]

# -- Default Env Settings ------------------------------------------------
DEFAULT_MAX_STEPS = 15
DEFAULT_SEED = 42
COLLABORATION_BONUS = 0.01
