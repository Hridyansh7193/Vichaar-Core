from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "name": "Software Update Rollout",
        "scenario": "Low conflict: A routine software update to improve system efficiency. Minor budget allocation needed but clear benefits for all.",
        "metrics": {
            "expected_profit": 0.5,
            "legal_risk": 0.1,
            "env_impact": 0.05,
            "public_sentiment": 0.6,
            "cost": 0.2
        },
        "entities": {"servers": 100, "dev_team": 5},
        "max_steps": 10
    },
    "medium": {
        "id": "medium",
        "name": "Personalized Ad Engine",
        "scenario": "Profit vs PR: Developing a new ad targeting engine using user behavioral data. High revenue potential but faces privacy legal hurdles.",
        "metrics": {
            "expected_profit": 0.7,
            "legal_risk": 0.4,
            "env_impact": 0.1,
            "public_sentiment": 0.4,
            "cost": 0.5
        },
        "entities": {"users": 1000, "ad_slots": 50},
        "max_steps": 15
    },
    "hard": {
        "id": "hard",
        "name": "Arctic Deep Mining",
        "scenario": "High Conflict: Deep-sea mining in the Arctic. Extreme profit potential vs devastating environmental impact and geopolitical tension.",
        "metrics": {
            "expected_profit": 0.9,
            "legal_risk": 0.8,
            "env_impact": 0.85,
            "public_sentiment": 0.2,
            "cost": 0.8
        },
        "entities": {"rigs": 2, "local_wildlife_index": 1.0},
        "max_steps": 20
    },
    "adversarial": {
        "id": "adversarial",
        "name": "Competitor Hostile Takeover",
        "scenario": "Survival Mode: A rival firm is attempting a hostile takeover. Actions must balance shareholder greed with long-term company stability and worker rights.",
        "metrics": {
            "expected_profit": 0.3,
            "legal_risk": 0.5,
            "env_impact": 0.2,
            "public_sentiment": 0.3,
            "cost": 0.6
        },
        "entities": {"shares_available": 1000000, "voting_power": 0.45},
        "max_steps": 20
    },
    "chaotic": {
        "id": "chaotic",
        "name": "Global Supply Chain Collapse",
        "scenario": "Uncertainty Max: Multiple overlapping crises (pandemic, war, climate event) have shattered supply chains. Every decision has unpredictable second-order effects.",
        "metrics": {
            "expected_profit": 0.1,
            "legal_risk": 0.6,
            "env_impact": 0.5,
            "public_sentiment": 0.1,
            "cost": 0.9
        },
        "entities": {"suppliers": 20, "warehouses": 5},
        "max_steps": 25
    }
}
