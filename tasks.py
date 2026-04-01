from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "scenario": "Low conflict: Simple software update to improve efficiency. Negligible risks.",
        "metrics": {
            "expected_profit": 0.5,
            "legal_risk": 0.1,
            "env_impact": 0.05,
            "public_sentiment": 0.6,
            "cost": 0.2
        }
    },
    "medium": {
        "id": "medium",
        "scenario": "Profit vs PR: Targeted ad campaign using detailed user data. High profit upside but significant privacy concerns.",
        "metrics": {
            "expected_profit": 0.7,
            "legal_risk": 0.4,
            "env_impact": 0.1,
            "public_sentiment": 0.4,
            "cost": 0.5
        }
    },
    "hard": {
        "id": "hard",
        "scenario": "High uncertainty and ethical conflict: Mining in a sensitive ecological region. Huge profit margins but severe environmental and legal risks.",
        "metrics": {
            "expected_profit": 0.9,
            "legal_risk": 0.8,
            "env_impact": 0.85,
            "public_sentiment": 0.2,
            "cost": 0.8
        }
    }
}
