"""
Grader — Final episode evaluation (separate from step reward).

Step reward  = delta-based per-agent signal (lives in env.py)
Final grade  = absolute state quality at end of episode (lives here)
"""

from typing import Dict, Any
from env_config import GRADE_WEIGHTS


def compute_final_grade(state: Dict[str, Any], task_id: str) -> float:
    """Score the final state of an episode on a 0–1 scale.
    
    Base formula:
        grade = Σ  GRADE_WEIGHTS[k] * metrics[k]
    
    Task-specific bonuses / penalties applied on top.
    """
    m = state.get("metrics", {})

    grade = sum(
        GRADE_WEIGHTS.get(k, 0.0) * float(m.get(k, 0.0))
        for k in GRADE_WEIGHTS
    )

    # ── task-specific adjustments ─────────────────────────────────────
    profit = float(m.get("expected_profit", 0.0))
    legal  = float(m.get("legal_risk", 0.0))
    env_   = float(m.get("env_impact", 0.0))
    sent   = float(m.get("public_sentiment", 0.0))

    if task_id == "easy":
        # bonus for keeping risks low (easy should be solved cleanly)
        if legal < 0.2 and env_ < 0.2:
            grade += 0.05

    elif task_id == "medium":
        # bonus for balanced outcome
        if profit > 0.5 and legal < 0.5 and sent > 0.3:
            grade += 0.05

    elif task_id == "hard":
        # severe penalty for ecological disaster
        if env_ > 0.7:
            grade -= 0.15
        # bonus for actually reducing impact
        if env_ < 0.4:
            grade += 0.10

    elif task_id == "adversarial":
        # company must survive
        if profit < 0.2:
            grade -= 0.10
        if profit > 0.5 and legal < 0.4:
            grade += 0.05

    elif task_id == "chaotic":
        # any stability is rewarded
        if profit > 0.3 and legal < 0.5:
            grade += 0.10
        if env_ > 0.7 or legal > 0.7:
            grade -= 0.10

    return round(max(0.0, min(1.0, float(grade))), 3)
