"""Grader -- final episode evaluation (separate from step reward)."""
from typing import Dict, Any
from configs.env_config import GRADE_WEIGHTS


def compute_final_grade(state: Dict[str, Any], task_id: str) -> float:
    m = state.get("metrics", {})

    grade = sum(
        GRADE_WEIGHTS.get(k, 0.0) * float(m.get(k, 0.0))
        for k in GRADE_WEIGHTS
    )

    profit = float(m.get("expected_profit", 0.0))
    legal  = float(m.get("legal_risk", 0.0))
    env_   = float(m.get("env_impact", 0.0))
    sent   = float(m.get("public_sentiment", 0.0))

    if task_id == "easy":
        if legal < 0.2 and env_ < 0.2:
            grade += 0.05
    elif task_id == "medium":
        if profit > 0.5 and legal < 0.5 and sent > 0.3:
            grade += 0.05
    elif task_id == "hard":
        if env_ > 0.7:
            grade -= 0.15
        if env_ < 0.4:
            grade += 0.10
    elif task_id == "adversarial":
        if profit < 0.2:
            grade -= 0.10
        if profit > 0.5 and legal < 0.4:
            grade += 0.05
    elif task_id == "chaotic":
        if profit > 0.3 and legal < 0.5:
            grade += 0.10
        if env_ > 0.7 or legal > 0.7:
            grade -= 0.10

    return round(max(0.0, min(1.0, float(grade))), 3)
