import random
from typing import Dict, Optional

SCENARIOS: Dict[str, str] = {
    "S1": "A local factory is releasing unverified emissions that might be harmful, but currently supports a low-cost profit model for the company.",
    "S2": "A sudden supply chain disruption requires switching to an unverified but cheaper supplier in a politically sensitive region.",
    "S3": "A new product has a small chance of failing safety checks; recalling it out of caution would cost millions, but ignoring it risks public trust and lawsuits.",
}

def get_scenario(scenario_id: Optional[str] = None) -> str:
    if scenario_id and scenario_id in SCENARIOS:
        return SCENARIOS[scenario_id]
    return random.choice(list(SCENARIOS.values()))
