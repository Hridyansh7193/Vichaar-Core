from typing import Dict, Any

def grade_episode(final_state: Dict[str, Any], task_id: str) -> float:
    m = final_state.get("metrics", {})
    
    score = (
        0.4 * float(m.get("expected_profit", 0.0))
        - 0.3 * float(m.get("legal_risk", 0.0))
        - 0.2 * float(m.get("env_impact", 0.0))
        + 0.1 * float(m.get("public_sentiment", 0.0))
    )

    # task-specific adjustments
    if task_id == "easy":
        score += 0.05 * (1 - float(m.get("legal_risk", 0.0)))

    elif task_id == "hard":
        score -= 0.05 * float(m.get("env_impact", 0.0))

    return round(max(0.0, min(1.0, score)), 2)
