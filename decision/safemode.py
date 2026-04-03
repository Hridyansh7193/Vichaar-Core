"""SafeMode -- detects score decline and bans risky actions."""
from typing import Dict, List
from decision.coordinator import compute_global_score
from configs.env_config import SAFE_MODE_SCORE_DECLINE


class SafeMode:
    """Activates when global score has been declining for 3+ steps."""

    def __init__(self):
        self._score_history: List[float] = []
        self.active = False

    def update(self, metrics: Dict[str, float]):
        score = compute_global_score(metrics)
        self._score_history.append(score)
        if len(self._score_history) >= 4:
            recent_delta = self._score_history[-1] - self._score_history[-3]
            self.active = recent_delta < -SAFE_MODE_SCORE_DECLINE
        else:
            self.active = False

    def reset(self):
        self._score_history.clear()
        self.active = False
