"""
Global Scoring -- compute_global_score used by Coordinator, CEO, SafeMode.

This module provides the single numeric objective the entire strategic
decision engine optimizes for. It is designed to correlate with the final
grade produced by evaluation.grader.compute_final_grade().
"""
from typing import Dict
from configs.env_config import GLOBAL_SCORE_WEIGHTS


def compute_global_score(metrics: Dict[str, float]) -> float:
    """Compute a single global quality score from current metrics.

    Formula:
        + 1.0 * expected_profit
        - 1.8 * legal_risk
        - 1.5 * cost
        - 1.3 * env_impact
        + 0.6 * public_sentiment

    Returns a float (can be negative for bad states).
    """
    return sum(
        GLOBAL_SCORE_WEIGHTS.get(k, 0.0) * metrics.get(k, 0.0)
        for k in GLOBAL_SCORE_WEIGHTS
    )
