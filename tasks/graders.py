from typing import Dict, Any

def compute_final_grade(state: Dict[str, Any], task_id: str) -> float:
    """Central grading logic based on metrics final state."""
    m = state.get("metrics", {})
    
    if task_id == "easy":
        score = (
          m.get("expected_profit", 0) +
          m.get("public_sentiment", 0) +
          (1 - m.get("legal_risk", 0))
        ) / 3
    elif task_id == "medium":
        score = (
          m.get("expected_profit", 0) +
          (1 - m.get("legal_risk", 0)) +
          m.get("public_sentiment", 0)
        ) / 3
    elif task_id == "hard":
        score = (
          m.get("expected_profit", 0) +
          (1 - m.get("env_impact", 0)) +
          (1 - m.get("legal_risk", 0))
        ) / 3
    elif task_id == "adversarial":
        score = (
          m.get("expected_profit", 0) +
          (1 - m.get("legal_risk", 0)) +
          m.get("public_sentiment", 0)
        ) / 3
    else: # chaotic
        score = (
          m.get("expected_profit", 0) +
          (1 - m.get("cost", 0)) +
          (1 - m.get("legal_risk", 0))
        ) / 3

    return float(max(0.0, min(1.0, score)))


def grade_easy(state: Dict[str, Any]) -> float:
    return compute_final_grade(state, "easy")


def grade_medium(state: Dict[str, Any]) -> float:
    return compute_final_grade(state, "medium")


def grade_hard(state: Dict[str, Any]) -> float:
    return compute_final_grade(state, "hard")


def grade_adversarial(state: Dict[str, Any]) -> float:
    return compute_final_grade(state, "adversarial")


def grade_chaotic(state: Dict[str, Any]) -> float:
    return compute_final_grade(state, "chaotic")
