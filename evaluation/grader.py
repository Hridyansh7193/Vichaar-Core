"""Grader -- final episode evaluation (separate from step reward).

Produces a score in [0.0, 1.0] that reflects how well the agents
navigated the scenario. Task-specific bonuses reward genuinely
difficult achievements (e.g., keeping env_impact low on Arctic Mining).
"""
from typing import Dict, Any
from configs.env_config import GRADE_WEIGHTS


def compute_final_grade(state: Dict[str, Any], task_id: str) -> float:
    """Compute final episode grade from terminal state metrics.

    Base grade uses GRADE_WEIGHTS; task-specific adjustments add
    bonuses/penalties for scenario-appropriate outcomes.
    """
    m = state.get("metrics", {})

    # Base weighted score
    grade = sum(
        GRADE_WEIGHTS.get(k, 0.0) * float(m.get(k, 0.0))
        for k in GRADE_WEIGHTS
    )

    profit = float(m.get("expected_profit", 0.0))
    legal  = float(m.get("legal_risk", 0.0))
    env_   = float(m.get("env_impact", 0.0))
    sent   = float(m.get("public_sentiment", 0.0))
    cost   = float(m.get("cost", 0.0))

    # -- Task-specific adjustments --
    if task_id == "easy":
        # Easy: reward clean execution with low risk
        if legal < 0.2 and env_ < 0.2:
            grade += 0.05
        if profit > 0.5 and cost < 0.4:
            grade += 0.03  # efficiency bonus

    elif task_id == "medium":
        # Medium: balanced profit + compliance + sentiment
        if profit > 0.5 and legal < 0.5 and sent > 0.3:
            grade += 0.05
        if cost < 0.6:
            grade += 0.02  # cost control bonus

    elif task_id == "hard":
        # Hard (Arctic Mining): penalize env destruction, reward mitigation
        if env_ > 0.7:
            grade -= 0.15
        if env_ < 0.4:
            grade += 0.10
        if legal < 0.4 and env_ < 0.5:
            grade += 0.05  # dual risk reduction

    elif task_id == "adversarial":
        # Adversarial (Hostile Takeover): must maintain profit + legal
        if profit < 0.2:
            grade -= 0.10
        if profit > 0.5 and legal < 0.4:
            grade += 0.05
        if sent > 0.4:
            grade += 0.03  # maintained public trust under siege

    elif task_id == "chaotic":
        # Chaotic (Supply Chain Collapse): survival is success
        if profit > 0.3 and legal < 0.5:
            grade += 0.10
        if env_ > 0.7 or legal > 0.7:
            grade -= 0.10
        if cost < 0.7:
            grade += 0.05  # cost control in chaos is impressive

    return round(max(0.0, min(1.0, float(grade))), 3)
