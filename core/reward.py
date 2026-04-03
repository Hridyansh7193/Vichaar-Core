"""Per-agent delta-based reward calculation."""
from typing import Dict
from configs.env_config import COLLABORATION_BONUS


def per_agent_rewards(
    prev: Dict[str, float],
    curr: Dict[str, float],
    collaborated: bool,
) -> Dict[str, float]:
    """Compute per-agent rewards from metric deltas."""
    def d(k: str) -> float:
        return curr.get(k, 0.0) - prev.get(k, 0.0)

    collab = COLLABORATION_BONUS if collaborated else 0.0

    return {
        "profit": round(d("expected_profit") * 2.0 - d("cost") + collab, 4),
        "ethics": round(-d("env_impact") * 1.5 - d("legal_risk") * 0.5 + collab, 4),
        "pr":     round(d("public_sentiment") * 2.0 + collab, 4),
        "legal":  round(-d("legal_risk") * 2.0 + collab, 4),
        "risk":   round(-(d("legal_risk") + d("env_impact") + d("cost")) + collab, 4),
    }
