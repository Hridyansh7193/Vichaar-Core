TASK = {
    "id": "task_easy",
    "name": "Software Update Rollout",
    "scenario": "Low conflict: A routine software update to improve system efficiency. Minor budget allocation needed but clear benefits for all.",
    "metrics": {
        "expected_profit": 0.5,
        "legal_risk": 0.1,
        "env_impact": 0.05,
        "public_sentiment": 0.6,
        "cost": 0.2,
    },
    "entities": {"servers": 100, "dev_team": 5},
    "max_steps": 10,
    "grader": "evaluation.grader:grade_easy",
}
