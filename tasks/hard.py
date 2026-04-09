TASK = {
    "id": "hard",
    "name": "Arctic Deep Mining",
    "scenario": "High Conflict: Deep-sea mining in the Arctic. Extreme profit potential vs devastating environmental impact and geopolitical tension.",
    "metrics": {
        "expected_profit": 0.9,
        "legal_risk": 0.8,
        "env_impact": 0.85,
        "public_sentiment": 0.2,
        "cost": 0.8,
    },
    "entities": {"rigs": 2, "local_wildlife_index": 1.0},
    "max_steps": 20,
    "grader": "evaluation.grader:grade_hard",
}
