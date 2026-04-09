TASK = {
    "id": "adversarial",
    "name": "Competitor Hostile Takeover",
    "scenario": "Survival Mode: A rival firm is attempting a hostile takeover. Actions must balance shareholder greed with long-term company stability and worker rights.",
    "metrics": {
        "expected_profit": 0.3,
        "legal_risk": 0.5,
        "env_impact": 0.2,
        "public_sentiment": 0.3,
        "cost": 0.6,
    },
    "entities": {"shares_available": 1000000, "voting_power": 0.45},
    "max_steps": 20,
    "grader": "evaluation.grader:compute_final_grade",
}
