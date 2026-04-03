"""
Vichaar-Core Environment -- Research-Grade Multi-Agent RL

OpenEnv API:  reset(task_id), step(action), state()
"""

import copy
import random as _random
from typing import Dict, Any, Tuple, List
from collections import Counter

from configs.env_config import (
    ACTIONS, ACTION_EFFECTS, EVENT_DEFS, PHASES,
    DEFAULT_MAX_STEPS, DEFAULT_SEED,
)
from core.reward import per_agent_rewards
from tasks import TASKS


class Env:
    """OpenEnv-compliant multi-agent RL environment."""

    def __init__(self, seed: int = DEFAULT_SEED):
        self._rng = _random.Random(seed)
        self.task_id: str = "medium"
        self.max_steps: int = DEFAULT_MAX_STEPS
        self._state: Dict[str, Any] = self._empty_state()

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "scenario": "Uninitialized",
            "phase": "morning",
            "metrics": {
                "expected_profit": 0.5,
                "legal_risk": 0.1,
                "env_impact": 0.05,
                "public_sentiment": 0.5,
                "cost": 0.0,
            },
            "entities": {},
            "events": [],
            "history": [],
            "step_count": 0,
            "agent_messages": [],
            "metrics_trend": [],
        }

    def state(self) -> Dict[str, Any]:
        return copy.deepcopy(self._state)

    def reset(self, task_id: str = "medium") -> Dict[str, Any]:
        if task_id not in TASKS:
            task_id = "medium"

        self.task_id = task_id
        task = TASKS[task_id]
        self.max_steps = task.get("max_steps", DEFAULT_MAX_STEPS)

        self._state = {
            "scenario": task.get("scenario", "Default scenario"),
            "phase": "morning",
            "metrics": task.get("metrics", {}).copy(),
            "entities": task.get("entities", {}).copy(),
            "events": [],
            "history": [],
            "step_count": 0,
            "agent_messages": [],
            "metrics_trend": [],
        }
        for key in ("expected_profit", "legal_risk", "env_impact", "public_sentiment", "cost"):
            self._state["metrics"].setdefault(
                key, 0.5 if key in ("expected_profit", "public_sentiment") else 0.0
            )
        return self.state()

    @staticmethod
    def _clamp(val: float) -> float:
        return round(max(0.0, min(1.0, float(val))), 3)

    def _phase_for(self, step: int) -> str:
        return PHASES[step % len(PHASES)]

    def _fire_events(self, metrics: Dict[str, float]) -> List[str]:
        triggered: List[str] = []
        for name, cfg in EVENT_DEFS.items():
            if self._rng.random() < cfg["prob"]:
                triggered.append(name)
                for m, delta in cfg["effects"].items():
                    metrics[m] = self._clamp(metrics.get(m, 0.0) + delta)
        return triggered

    @staticmethod
    def _detect_collaboration(agent_votes: Dict[str, str]) -> bool:
        if not agent_votes:
            return False
        counts = Counter(agent_votes.values())
        return counts.most_common(1)[0][1] >= 3

    def step(
        self,
        action: str,
        messages: List[str] | None = None,
        agent_votes: Dict[str, str] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, float], bool, Dict[str, Any]]:
        if messages is None:
            messages = []
        if agent_votes is None:
            agent_votes = {}

        prev_metrics = copy.deepcopy(self._state["metrics"])
        metrics = self._state["metrics"]

        if action in ACTION_EFFECTS:
            for m, delta in ACTION_EFFECTS[action].items():
                metrics[m] = metrics.get(m, 0.0) + delta

        for k in list(metrics.keys()):
            metrics[k] = self._clamp(metrics[k])

        events = self._fire_events(metrics)

        self._state["events"] = events
        self._state["history"].append(action)
        self._state["step_count"] += 1
        self._state["phase"] = self._phase_for(self._state["step_count"])
        self._state["agent_messages"] = messages
        self._state["metrics_trend"].append(copy.deepcopy(metrics))

        collaborated = self._detect_collaboration(agent_votes)
        rewards = per_agent_rewards(prev_metrics, metrics, collaborated)

        done = self._state["step_count"] >= self.max_steps
        info = {
            "events": events,
            "collaborated": collaborated,
            "agent_votes": agent_votes,
        }
        return self.state(), rewards, done, info
