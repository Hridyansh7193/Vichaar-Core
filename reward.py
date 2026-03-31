from typing import Dict, Any, Optional, List

def evaluate_decision(
    final_decision: Dict[str, Any],
    state_metrics: Dict[str, float],
    ground_truth: Optional[Dict[str, Any]] = None,
    agent_outputs: Optional[List[Dict[str, Any]]] = None
) -> float:
    """
    Evaluates the final decision and returns a reward score between -1.0 and 1.0.
    
    state_metrics is expected to contain normalized values (0.0 to 1.0):
    - profit: 0.0 (total loss) to 1.0 (max profit)
    - legal_risk: 0.0 (no risk) to 1.0 (extreme risk)
    - pr_risk: 0.0 (no risk) to 1.0 (disaster)
    - env_risk: 0.0 (no risk) to 1.0 (high impact)
    
    +1.0 -> well-balanced optimal decision
    0.0  -> neutral / acceptable
    Negative -> bad decisions
    """
    profit = state_metrics.get("profit", 0.0)
    legal_risk = state_metrics.get("legal_risk", 0.0)
    pr_risk = state_metrics.get("pr_risk", 0.0)
    env_risk = state_metrics.get("env_risk", 0.0)
 
    average_risk = (legal_risk + pr_risk + env_risk) / 3.0
    reward = profit - average_risk
    
    max_risk = max(legal_risk, pr_risk, env_risk)

    if max_risk >= 0.7:
        reward -= 0.5 
        if profit >= 0.7:

            reward -= 0.5 


    if profit <= 0.1 and max_risk <= 0.2:
        reward -= 0.4 

   
    if 0.4 <= profit <= 0.8 and max_risk < 0.4:
         reward += 0.3
         
    return max(-1.0, min(1.0, reward))
