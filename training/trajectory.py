"""
Trajectory Collector -- research-grade JSONL logging.

Stores per-step transitions with full provenance:
  state, action, agent_votes, rewards, next_state, events, step, timestamp
"""

import json
import os
import time
from typing import Dict, Any, List


class TrajectoryCollector:
    def __init__(self, output_dir: str = "trajectories"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._buffer: List[Dict[str, Any]] = []

    def log_step(
        self,
        step: int,
        state: Dict[str, Any],
        action: str,
        agent_votes: Dict[str, str],
        rewards: Dict[str, float],
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ):
        self._buffer.append({
            "step": step,
            "timestamp": time.time(),
            "state_metrics": state.get("metrics", {}),
            "action": action,
            "agent_votes": agent_votes,
            "rewards": rewards,
            "next_state_metrics": next_state.get("metrics", {}),
            "events": info.get("events", []),
            "collaborated": info.get("collaborated", False),
        })

    def save_episode(self, episode_id: str):
        path = os.path.join(self.output_dir, f"episode_{episode_id}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        count = len(self._buffer)
        self._buffer.clear()
        return count

    def reset(self):
        self._buffer.clear()
