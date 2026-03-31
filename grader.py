from typing import Dict, Any, Optional
import config

def compute_reward(state: Dict[str, Any], final_output: Dict[str, Any], task_id: str) -> float:
    """
    Computes a final reward score (0.0 to 1.0) based on task-specific criteria.
    """
    metrics = state.get("metrics", config.DEFAULT_METRICS)
    decision = final_output.get("final_decision", "modify")
    
    # 1. Normalize Profit (0–1)
    # Using task-specific caps might be better, but assuming 1M/Scenario
    # For 'hard' task, the profit target is much larger (200M)
    profit_cap = 1000000.0
    if task_id == "hard":
        profit_cap = 200000000.0
        
    expected_profit = metrics.get("expected_profit", 0.0)
    profit_score = min(expected_profit / profit_cap, 1.0)
    
    # 2. Extract Risks and Sentiment
    legal_risk = metrics.get("legal_risk", 0.0)
    env_impact = metrics.get("env_impact", 0.0)
    public_sentiment = metrics.get("public_sentiment", 0.0)
    
    # 3. Base Weighted Score (Goal for 0.0-1.0)
    # Weights sum to approx 0 if risks are high, etc.
    # We calibrate to return somewhere near 0.5 for neutral
    base_score = (
        (0.4 * profit_score)
        - (0.3 * legal_risk)
        - (0.2 * env_impact)
        + (0.1 * (public_sentiment + 1.0) / 2.0) # Map sentiment [-1,1] to [0,1]
    )
    
    # 4. Task-Specific Strictness Logic
    if task_id == "easy":
        # Any 'launch' with measurable risk on an easy task is a fail
        if decision == "launch" and (legal_risk > 0.1 or env_impact > 0.1):
            base_score -= 0.5
            
    elif task_id == "medium":
        # Balanced trade-offs are expected
        if decision == "modify":
            base_score += 0.1
            
    elif task_id == "hard":
        # Heavy risk awareness is mandatory
        if decision == "launch" and (legal_risk > 0.5 or env_impact > 0.5):
            base_score -= 0.6
        if decision == "reject" and (legal_risk > 0.7 or env_impact > 0.7):
            base_score += 0.2

    # Map back to 0.0-1.0 range
    final_reward = (base_score + 1.0) / 2.0 # Normalizing mapping
    return max(0.0, min(1.0, float(final_reward)))

def get_intermediate_reward(state: Dict[str, Any], round_num: int) -> float:
    """Give small positive rewards for successful agent deliberations."""
    history = state.get("history", [])
    if not history:
        return 0.0
        
    # Check if agents in this round provided valid info (not fallbacks)
    last_actions = history[-3:] # Assume 3 agents per round (Profit, PR, Ethics)
    confidence_sum = sum(a.get("confidence", 0.0) for a in last_actions)
    
    # Reward for high confidence/clarity in deliberation
    return min(0.05, (confidence_sum / 3.0) * 0.05) if confidence_sum > 0 else 0.0
