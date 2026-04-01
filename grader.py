from typing import Dict, Any

def compute_reward(state: Dict[str, Any], task_id: str) -> float:
    """
    Computes reward based on the specified formula:
    + 0.4 * profit - 0.3 * legal_risk - 0.2 * env_impact + 0.1 * sentiment
    Normalized to 0..1 scale. Task-specific logic adjustments.
    """
    metrics = state.get("metrics", {})

    
    profit = float(metrics.get("expected_profit", 0.0))
    legal_risk = float(metrics.get("legal_risk", 0.0))
    env_impact = float(metrics.get("env_impact", 0.0))
    sentiment = float(metrics.get("public_sentiment", 0.0))
    
    # Formula calculates a raw score
    reward = (0.4 * profit) - (0.3 * legal_risk) - (0.2 * env_impact) + (0.1 * sentiment)
    
    # Task specific logic
    if task_id == "easy":
        # Strict logic: punish any severe risk buildup since issue is simple 
        if legal_risk > 0.3 or env_impact > 0.3:
            reward -= 0.3
            
    elif task_id == "medium":
        # Balanced: Reward finding a middle ground (low to moderate impact/risk with decent profit)
        if profit > 0.5 and legal_risk < 0.5 and sentiment > 0.3:
            reward += 0.1
            
    elif task_id == "hard":
        # Risk-aware: severely penalize out-of-control risks, as they threaten the company's existence
        if legal_risk > 0.7 or env_impact > 0.7:
            reward -= 0.4

    # Ensure final reward stays strictly clamped between 0.05 and 1.0
    return round(max(0.05, min(1.0, float(reward))), 2)
