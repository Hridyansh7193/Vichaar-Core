from typing import Dict, Any
import config

def compute_reward(state: Dict[str, Any], final_output: Dict[str, Any]) -> float:
    """
    Computes a normalized reward score between -1.0 and 1.0.
    
    This version uses weighted scoring and decision-based adjustments to
    favor balanced, low-risk business decisions.
    """
    metrics = state.get("metrics", config.DEFAULT_METRICS)
    decision = final_output.get("final_decision", "modify")
    
    # 1. Normalize Profit (Cap at 1M)
    expected_profit = metrics.get("expected_profit", 0.0)
    profit_score = min(expected_profit / config.PROFIT_CAP, 1.0)
    
    # 2. Extract Risks and Sentiment (Assumed normalized 0-1)
    legal_risk = metrics.get("legal_risk", 0.0)
    env_impact = metrics.get("env_impact", 0.0)
    public_sentiment = metrics.get("public_sentiment", 0.0)
    
    # 3. Base Weighted Score (Interpretable and Deterministic)
    # The weights define the priority of the system
    reward = (
        (config.PROFIT_WEIGHT * profit_score)
        - (config.LEGAL_RISK_WEIGHT * legal_risk)
        - (config.ENV_IMPACT_WEIGHT * env_impact)
        + (config.SENTIMENT_WEIGHT * (max(0.0, public_sentiment))) # Positive sentiment helps
    )
    
    # 4. Decision-Based Adjustments
    # High Risk Penalty
    high_risks = (legal_risk > config.RISK_THRESHOLD or env_impact > config.RISK_THRESHOLD)
    
    if decision == "launch" and high_risks:
        # Aggressive launching when risks are extreme is heavily discouraged
        reward -= config.LAUNCH_PENALTY_VALUE
        
    elif decision == "modify":
        # Modifications that balance priorities are encouraged
        reward += config.MODIFY_BONUS_VALUE
        
    elif decision == "reject" and high_risks:
        # Rejecting a high-profit but high-risk project is a responsible action
        reward += config.REJECT_REWARD_VALUE
        
    # 5. Output Normalization (Clamped to [-1, 1])
    reward = max(-1.0, min(1.0, reward))
    
    return round(float(reward), 3)
