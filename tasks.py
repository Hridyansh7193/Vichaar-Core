from typing import Dict, Any

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "id": "easy",
        "description": "Clear-cut case of process optimization with zero significant risks.",
        "scenario": "Implementing a new software update that reduces server energy consumption by 15% with zero downtime. Customer data is handled strictly locally.",
        "metrics": {
            "cost": 10000.0,
            "expected_profit": 200000.0,
            "legal_risk": 0.05,
            "env_impact": 0.0,
            "public_sentiment": 0.8
        },
        "uncertainty": {"legal_conf": 0.99, "env_conf": 1.0}
    },
    "medium": {
        "id": "medium",
        "description": "Marketing feature that boosts profit but carries public relations risks.",
        "scenario": "Using personalized social media data to target high-frequency shoppers. It increases engagement by 40% but might be seen as intrusive by privacy advocates.",
        "metrics": {
            "cost": 75000.0,
            "expected_profit": 800000.0,
            "legal_risk": 0.3,
            "env_impact": 0.1,
            "public_sentiment": -0.4
        },
        "uncertainty": {"legal_conf": 0.8, "env_conf": 0.9}
    },
    "hard": {
        "id": "hard",
        "description": "High-stakes decision involving conflicting ethical and legal mandates.",
        "scenario": "A potential lithium mining project in a sensitive region. It could secure the company's battery supply for 10 years and boost profits by $200M, but involves displacement of local communities and high legal uncertainty regarding tribal land rights.",
        "metrics": {
            "cost": 5000000.0,
            "expected_profit": 200000000.0,
            "legal_risk": 0.8,
            "env_impact": 0.95,
            "public_sentiment": -0.85
        },
        "uncertainty": {"legal_conf": 0.4, "env_conf": 0.5}
    }
}
