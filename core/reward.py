"""Per-agent delta-based reward calculation with improved shaping."""
from typing import Dict
from configs.env_config import COLLABORATION_BONUS


def per_agent_rewards(
    prev: Dict[str, float],
    curr: Dict[str, float],
    collaborated: bool,
) -> Dict[str, float]:
    """Compute per-agent rewards from metric deltas.

    Improvement over v1: added cross-metric awareness so agents get
    small penalties for side-effects outside their primary concern,
    encouraging genuinely balanced decision-making.
    """
    def d(k: str) -> float:
        return curr.get(k, 0.0) - prev.get(k, 0.0)

    collab = COLLABORATION_BONUS if collaborated else 0.0

    # Base rewards
    profit_reward = d("expected_profit") * 2.0 - d("cost")
    ethics_reward = -d("env_impact") * 1.5 - d("legal_risk") * 0.5 + max(0, d("public_sentiment")) * 0.2
    pr_reward = d("public_sentiment") * 2.0
    legal_reward = -d("legal_risk") * 2.0
    risk_reward = -(d("legal_risk") + d("env_impact") + d("cost"))

    raw_rewards = {
        "profit": profit_reward,
        "ethics": ethics_reward,
        "pr": pr_reward,
        "legal": legal_reward,
        "risk": risk_reward,
    }

    # 1. Downside Penalty: Stronger punishment for negative drops
    penalty_scale = 1.5
    for k in raw_rewards:
        if raw_rewards[k] < 0:
            raw_rewards[k] *= penalty_scale

    # 2. Stability Bonus
    if abs(d("expected_profit")) < 0.05 and abs(d("legal_risk")) < 0.05:
        for k in raw_rewards:
            raw_rewards[k] += 0.05

    # 3. Critical Threshold Penalty (bad state zones)
    if curr.get("expected_profit", 0) < 0.2 or curr.get("public_sentiment", 0) < 0.2:
        for k in raw_rewards:
            raw_rewards[k] -= 0.1

    # 4. Anti-Greedy Control (Prevent profit over-optimization at cost of risk)
    if d("expected_profit") > 0 and d("legal_risk") > 0:
        raw_rewards["profit"] -= d("legal_risk") * 2.0

    return {k: round(v + collab, 4) for k, v in raw_rewards.items()}
