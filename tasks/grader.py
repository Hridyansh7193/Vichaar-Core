from typing import Dict, Any

def compute_final_grade(observation: Dict[str, Any], *args, **kwargs) -> float:
    """Central grading logic compatible with multiple caller signatures."""
    if not observation:
        return 0.0
        
    # Standard OpenEnv observation often wraps state in an 'observation' key
    # or passes the metrics directly. We handle both.
    m = observation.get("metrics", observation)
    
    # Simple weighted score (fallback)
    score = (
        m.get("expected_profit", 0) + 
        m.get("public_sentiment", 0) + 
        (1 - m.get("legal_risk", 0))
    ) / 3
    
    return float(max(0.0, min(1.0, score)))


def grade_easy(observation, *args, **kwargs) -> float:
    return compute_final_grade(observation, *args, **kwargs)


def grade_medium(observation, *args, **kwargs) -> float:
    return compute_final_grade(observation, *args, **kwargs)


def grade_hard(observation, *args, **kwargs) -> float:
    return compute_final_grade(observation, *args, **kwargs)
