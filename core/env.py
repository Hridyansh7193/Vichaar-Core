"""
Vichaar-Core Environment -- Research-Grade Multi-Agent RL

OpenEnv API:  reset(task_id), step(action), state(), close()
"""

import copy
import random as _random
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
import gymnasium as gym
from gymnasium import spaces

from configs.env_config import (
    ACTIONS, ACTION_EFFECTS, EVENT_DEFS, PHASES,
    DEFAULT_MAX_STEPS, DEFAULT_SEED,
)
from core.reward import per_agent_rewards
from tasks import TASKS


class Env(gym.Env):
    """OpenEnv-compliant multi-agent RL environment."""

    def __init__(self, seed: int = DEFAULT_SEED, deterministic: bool = True):
        super().__init__()
        self._seed = seed
        self._rng = _random.Random(seed)
        self.deterministic: bool = deterministic
        self.task_id: str = "easy"
        self.max_steps: int = DEFAULT_MAX_STEPS

        # Gymnasium spaces
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Dict({
            "expected_profit": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "legal_risk": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "env_impact": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "public_sentiment": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
            "cost": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=float),
        })

        self._state: Dict[str, Any] = self._empty_state()
        self._closed: bool = False

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

    def reset(self, task_id: str = "easy", **kwargs) -> Dict[str, Any]:
        self._closed = False

        # Look up task in registry, fall back to easy
        task_def = TASKS.get(task_id, TASKS.get("easy"))
        self.task_id = task_def["id"]
        self.max_steps = task_def.get("max_steps", DEFAULT_MAX_STEPS)

        # Reset state from task definition
        self._state = self._empty_state()
        self._state["scenario"] = task_def.get("scenario", "Uninitialized")
        self._state["metrics"] = copy.deepcopy(task_def.get("metrics", self._state["metrics"]))
        self._state["entities"] = copy.deepcopy(task_def.get("entities", {}))

        return self.state()

    @staticmethod
    def _clamp(val: float) -> float:
        return round(max(0.0, min(1.0, float(val))), 3)

    def _phase_for(self, step: int) -> str:
        return PHASES[step % len(PHASES)]

    def _fire_events(self, metrics: Dict[str, float]) -> List[str]:
        triggered: List[str] = []
        for name, cfg in EVENT_DEFS.items():
            if not self.deterministic and self._rng.random() < cfg["prob"]:
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
        messages: Optional[List[str]] = None,
        agent_votes: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
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
        multi_agent_rewards = per_agent_rewards(prev_metrics, metrics, collaborated)

        # Compute Global Reward
        global_reward = float(sum(multi_agent_rewards.values()) / len(multi_agent_rewards))

        # Action Uniqueness Bonus
        unique_actions = len(set(self._state["history"][-4:])) if self._state["history"] else 1
        global_reward += 0.02 * unique_actions

        # Strong Anti-Loop Penalty (generic — ANY repeated action)
        if len(self._state["history"]) >= 3:
            if self._state["history"][-3:].count(action) >= 2:
                global_reward -= 0.15

        # Balanced Metrics Reward
        metric_vals = list(metrics.values())
        if metric_vals:
            spread = max(metric_vals) - min(metric_vals)
            global_reward += (1.0 - spread) * 0.05

        # Extreme State Penalties
        if metrics.get("legal_risk", 0) > 0.8:
            global_reward -= 0.2
        if metrics.get("cost", 0) > 0.9:
            global_reward -= 0.2
        if metrics.get("env_impact", 0) > 0.8:
            global_reward -= 0.2

        # Boosted Collaboration Bonus
        if collaborated:
            global_reward += 0.1

        if self.task_id == "chaotic":
            global_reward += 0.05

        # Final Clamp
        global_reward = max(0.0, min(1.0, global_reward))

        # Early Termination for Critical Failures
        done = self._state["step_count"] >= self.max_steps
        if (
            metrics.get("cost", 0) >= 0.98 or
            metrics.get("legal_risk", 0) >= 0.98 or
            metrics.get("env_impact", 0) >= 0.98
        ):
            done = True
            global_reward = 0.0

        info = {
            "events": events,
            "collaborated": collaborated,
            "agent_votes": agent_votes,
            "rewards": multi_agent_rewards,
        }
        return self.state(), global_reward, done, info

    def close(self):
        """Graceful shutdown — required by OpenEnv spec."""
        self._closed = True
