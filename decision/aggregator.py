"""Aggregator -- board parsing and vote counting utilities."""
from typing import Dict, List
from configs.env_config import ACTIONS


def parse_board_suggestions(board: List[str]) -> Dict[str, int]:
    """Count action mentions in discussion board messages."""
    counts: Dict[str, int] = {}
    combined = " ".join(board).lower()
    for act in ACTIONS:
        act_search = act.replace("_", " ")
        n = combined.count(act) + combined.count(act_search)
        if n > 0:
            counts[act] = n
    return counts
