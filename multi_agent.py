"""
Multi-Agent System -- Memory, Discussion, Voting, Reflection

Architecture per step:
  1. Discussion  -- agents post proposals to a shared board
  2. Voting      -- each agent scores ALL actions using state + memory + learning
  3. Aggregation -- majority vote with collaboration detection
  4. Memory      -- store (state, action, reward) with importance scoring
  5. Reflection  -- periodic strategy summary

Decision engine (heuristic mode):
  * State-aware urgency scoring (react to metric levels & trends)
  * Phase-based strategy (explore/optimize/correct/plan)
  * Learned action values (updated from rewards, alpha=0.3)
  * Repetition kill switch (penalize recent repeats)
  * Epsilon-greedy + softmax exploration
  * Discussion agreement bonus (drives collaboration)
  * Memory-driven failure pattern detection & avoidance
  * Diversity bonus for underused actions
  * Explainable reasoning for every decision
"""

import json
import math
import asyncio
import logging
import copy
import random as _py_random
from typing import Dict, Any, List, Tuple
from collections import Counter

from openai import AsyncOpenAI

import config
from env_config import ACTIONS, AGENT_DEFS, ACTION_EFFECTS, MEMORY_CAPACITY

logger = logging.getLogger(__name__)

# -- Hyperparameters ---------------------------------------------------
ALPHA = 0.3
BASE_EPSILON = 0.15
CRISIS_EPSILON = 0.4
SOFTMAX_TEMP = 0.5
REPEAT_PENALTY = 0.3
AGREEMENT_BONUS = 0.2
DIVERSITY_BONUS = 0.1
MEMORY_AVOIDANCE = 0.15

# Phase strategy modifiers
PHASE_STRATEGY = {
    "morning":   {"epsilon_mult": 1.4, "label": "EXPLORE",  "diversity_mult": 2.0, "repeat_mult": 0.5},
    "execution": {"epsilon_mult": 0.5, "label": "OPTIMIZE", "diversity_mult": 0.5, "repeat_mult": 1.0},
    "review":    {"epsilon_mult": 0.8, "label": "CORRECT",  "diversity_mult": 1.0, "repeat_mult": 1.5},
    "planning":  {"epsilon_mult": 1.0, "label": "PLAN",     "diversity_mult": 1.5, "repeat_mult": 0.8},
}

# Metric display names for readable explanations
METRIC_LABELS = {
    "expected_profit": "profit",
    "legal_risk": "legal risk",
    "env_impact": "env impact",
    "public_sentiment": "sentiment",
    "cost": "cost",
}


# ======================================================================
#  Memory Stream
# ======================================================================
class MemoryStream:
    """Per-agent episodic memory with importance-based eviction."""

    def __init__(self, capacity: int = MEMORY_CAPACITY):
        self.capacity = capacity
        self.memories: List[Dict[str, Any]] = []
        self.reflections: List[str] = []

    def add(self, metrics_snapshot: Dict[str, float], action: str, reward: float, step: int):
        entry = {
            "metrics": metrics_snapshot,
            "action": action,
            "reward": reward,
            "step": step,
            "importance": round(abs(reward) * 10, 2),
        }
        self.memories.append(entry)
        if len(self.memories) > self.capacity:
            self.memories.sort(key=lambda x: x["importance"])
            self.memories.pop(0)

    def recent(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.memories[-n:]

    def recent_actions(self, n: int = 3) -> List[str]:
        return [m["action"] for m in self.memories[-n:]]

    def recent_rewards(self, n: int = 3) -> List[float]:
        return [m["reward"] for m in self.memories[-n:]]

    def avg_reward_for_action(self, action: str) -> float:
        matching = [m["reward"] for m in self.memories if m["action"] == action]
        return sum(matching) / len(matching) if matching else 0.0

    def actions_used_recently(self, n: int = 5) -> set:
        return set(m["action"] for m in self.memories[-n:])

    def detect_failure_pattern(self) -> str | None:
        """Detect if recent actions led to consistently negative rewards."""
        recent = self.recent(4)
        if len(recent) < 3:
            return None
        neg_count = sum(1 for m in recent if m["reward"] < -0.02)
        if neg_count >= 3:
            # Identify the most repeated failing action
            fail_actions = [m["action"] for m in recent if m["reward"] < -0.02]
            if fail_actions:
                counts = Counter(fail_actions)
                worst = counts.most_common(1)[0]
                if worst[1] >= 2:
                    return worst[0]
        return None

    def add_reflection(self, text: str):
        self.reflections.append(text)
        if len(self.reflections) > 10:
            self.reflections.pop(0)

    def latest_reflection(self) -> str:
        return self.reflections[-1] if self.reflections else "No prior reflection."

    def clear(self):
        self.memories.clear()
        self.reflections.clear()


# ======================================================================
#  Agent
# ======================================================================
class Agent:
    """Single agent with persona, learned action values, and explainable reasoning."""

    def __init__(self, role: str, definition: Dict[str, Any]):
        self.role = role
        self.desc = definition["desc"]
        self.reward_weights = definition["reward_weights"]
        self.preferred_actions: List[str] = definition["preferred_actions"]
        self.memory = MemoryStream()
        self.action_values: Dict[str, float] = {a: 0.0 for a in ACTIONS}
        self._last_reason: str = ""

        seed_offset = hash(role) % 10000
        self._rng = _py_random.Random(42 + seed_offset)

        try:
            self.client = AsyncOpenAI(
                api_key=config.OPENAI_API_KEY or "dummy",
                base_url=config.OPENAI_API_BASE,
            )
        except Exception:
            self.client = None

    # -- update action values from reward feedback ---------------------
    def update_values(self, action: str, reward: float):
        if action in self.action_values:
            old = self.action_values[action]
            self.action_values[action] = old + ALPHA * (reward - old)

    def top_values(self, n: int = 3) -> List[Tuple[str, float]]:
        """Return top N learned action values for visibility."""
        return sorted(self.action_values.items(), key=lambda x: -x[1])[:n]

    # -- identify most urgent metric for this agent --------------------
    def _find_trigger_metric(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        """Return (metric_name, human_reason) for the metric most demanding attention."""
        worst_metric = None
        worst_urgency = -999.0
        for m, w in self.reward_weights.items():
            val = metrics.get(m, 0.5)
            # How urgent is this metric for this agent?
            if w < 0:  # agent wants it LOW
                urgency = val * abs(w)  # high value = bad
            else:       # agent wants it HIGH
                urgency = (1.0 - val) * w  # low value = bad
            if urgency > worst_urgency:
                worst_urgency = urgency
                worst_metric = m
        label = METRIC_LABELS.get(worst_metric, worst_metric)
        val = metrics.get(worst_metric, 0.5)
        direction = "too high" if self.reward_weights.get(worst_metric, 0) < 0 else "too low"
        return worst_metric, f"{label} is {direction} ({val:.2f})"

    # -- state-aware urgency scoring -----------------------------------
    def _urgency_scores(self, metrics: Dict[str, float]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for act in ACTIONS:
            effects = ACTION_EFFECTS.get(act, {})
            score = 0.0
            for metric, delta in effects.items():
                current = metrics.get(metric, 0.5)
                if metric in ("legal_risk", "env_impact", "cost"):
                    if current > 0.6:
                        score += (-delta) * current * 2.0
                    elif current > 0.3:
                        score += (-delta) * 0.5
                    else:
                        score += (-delta) * 0.1
                elif metric in ("expected_profit", "public_sentiment"):
                    if current < 0.4:
                        score += delta * (1.0 - current) * 2.0
                    elif current < 0.7:
                        score += delta * 0.5
                    else:
                        score += delta * 0.1
            for metric, weight in self.reward_weights.items():
                if metric in effects:
                    score += effects[metric] * weight * 0.5
            if act in self.preferred_actions:
                score += 0.05
            scores[act] = score
        return scores

    # -- full scoring pipeline with phase strategy ---------------------
    def _score_actions(
        self,
        state: Dict[str, Any],
        board_suggestions: Dict[str, int],
    ) -> Dict[str, float]:
        metrics = state.get("metrics", {})
        phase = state.get("phase", "execution")
        phase_cfg = PHASE_STRATEGY.get(phase, PHASE_STRATEGY["execution"])

        urgency = self._urgency_scores(metrics)
        recent_acts = self.memory.recent_actions(3)
        used_recently = self.memory.actions_used_recently(5)
        failure_action = self.memory.detect_failure_pattern()

        scores: Dict[str, float] = {}
        for act in ACTIONS:
            s = urgency.get(act, 0.0)
            s += self.action_values.get(act, 0.0)

            # Repetition kill (phase-modulated)
            repeat_count = recent_acts.count(act)
            if repeat_count >= 2:
                s -= REPEAT_PENALTY * repeat_count * phase_cfg["repeat_mult"]
            elif repeat_count == 1:
                s -= REPEAT_PENALTY * 0.5 * phase_cfg["repeat_mult"]

            # Discussion agreement
            agreement = board_suggestions.get(act, 0)
            if agreement > 0:
                s += AGREEMENT_BONUS * agreement

            # Diversity (phase-modulated)
            if act not in used_recently and len(used_recently) > 0:
                s += DIVERSITY_BONUS * phase_cfg["diversity_mult"]

            # Memory-driven avoidance
            avg_r = self.memory.avg_reward_for_action(act)
            if avg_r < -0.05:
                s -= MEMORY_AVOIDANCE

            # Strategic failure avoidance
            if failure_action and act == failure_action:
                s -= 0.5  # hard penalty for detected failure pattern

            scores[act] = s
        return scores

    # -- softmax sampling ----------------------------------------------
    def _softmax_sample(self, scores: Dict[str, float]) -> str:
        actions = list(scores.keys())
        values = [scores[a] / SOFTMAX_TEMP for a in actions]
        max_v = max(values)
        exps = [math.exp(v - max_v) for v in values]
        total = sum(exps)
        probs = [e / total for e in exps]
        return self._rng.choices(actions, weights=probs, k=1)[0]

    # -- epsilon computation (phase-aware) -----------------------------
    def _get_epsilon(self, phase: str = "execution") -> float:
        phase_cfg = PHASE_STRATEGY.get(phase, PHASE_STRATEGY["execution"])
        recent_r = self.memory.recent_rewards(3)
        base = BASE_EPSILON
        if len(recent_r) >= 2 and sum(recent_r) / len(recent_r) < 0:
            base = CRISIS_EPSILON
        return min(0.5, base * phase_cfg["epsilon_mult"])

    # -- main heuristic action with explanation ------------------------
    def _heuristic_action(
        self,
        state: Dict[str, Any],
        board_suggestions: Dict[str, int] | None = None,
    ) -> str:
        if board_suggestions is None:
            board_suggestions = {}

        metrics = state.get("metrics", {})
        phase = state.get("phase", "execution")
        scores = self._score_actions(state, board_suggestions)
        eps = self._get_epsilon(phase)

        trigger_metric, trigger_reason = self._find_trigger_metric(metrics)
        failure_action = self.memory.detect_failure_pattern()

        if self._rng.random() < eps:
            chosen = self._rng.choice(ACTIONS)
            phase_label = PHASE_STRATEGY.get(phase, {}).get("label", "?")
            self._last_reason = f"[{phase_label}] Exploring: {trigger_reason}"
        else:
            chosen = self._softmax_sample(scores)
            phase_label = PHASE_STRATEGY.get(phase, {}).get("label", "?")
            # Build explanation
            parts = [f"[{phase_label}]"]
            parts.append(f"Trigger: {trigger_reason}")
            if failure_action:
                parts.append(f"Avoiding failed: {failure_action}")
            if board_suggestions.get(chosen, 0) > 0:
                parts.append(f"Board agreed ({board_suggestions[chosen]}x)")
            self._last_reason = " | ".join(parts)

        return chosen

    @property
    def last_reason(self) -> str:
        return self._last_reason

    # -- heuristic discussion message ----------------------------------
    def _heuristic_message(self, state: Dict[str, Any]) -> str:
        metrics = state.get("metrics", {})
        _, trigger_reason = self._find_trigger_metric(metrics)
        urgency = self._urgency_scores(metrics)
        best_act = max(urgency, key=urgency.get)
        return f"[{self.role}] {trigger_reason}. I propose {best_act}."

    # -- discuss -------------------------------------------------------
    async def discuss(self, state: Dict[str, Any], board: List[str]) -> str:
        if not config.OPENAI_API_KEY or not self.client:
            return self._heuristic_message(state)
        prompt = (
            f"You are the {self.role} agent. Goal: {self.desc}\n"
            f"Board so far: {board}\n"
            f"State metrics: {json.dumps(state.get('metrics', {}))}\n"
            f"Reflection: {self.memory.latest_reflection()}\n\n"
            "Write ONE sentence proposing your preferred next action and why."
        )
        try:
            resp = await self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2, max_tokens=80,
            )
            return f"[{self.role}] {resp.choices[0].message.content.strip()}"
        except Exception:
            return self._heuristic_message(state)

    # -- vote ----------------------------------------------------------
    async def vote(
        self,
        state: Dict[str, Any],
        board: List[str],
        board_suggestions: Dict[str, int] | None = None,
    ) -> str:
        if not config.OPENAI_API_KEY or not self.client:
            return self._heuristic_action(state, board_suggestions)
        prompt = (
            f"You are the {self.role} agent.\nPersona: {self.desc}\n"
            f"Discussion Board:\n" + "\n".join(board) + "\n"
            f"Metrics: {json.dumps(state.get('metrics', {}), indent=2)}\n"
            f"Memory: {json.dumps(self.memory.recent(3), indent=2)}\n"
            f"Valid actions: {ACTIONS}\n\n"
            'Return JSON: {"action": "<exact_choice>"}'
        )
        try:
            resp = await self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=60,
            )
            data = json.loads(resp.choices[0].message.content)
            action = data.get("action", "")
            if action in ACTIONS:
                self._last_reason = f"LLM selected based on board consensus"
                return action
        except Exception:
            pass
        return self._heuristic_action(state, board_suggestions)

    # -- reflect with failure detection --------------------------------
    async def reflect(self, state: Dict[str, Any]):
        recent = self.memory.recent(5)
        if not recent:
            return
        if not config.OPENAI_API_KEY or not self.client:
            avg_r = sum(m["reward"] for m in recent) / len(recent)
            action_counts = Counter(m["action"] for m in recent)
            most_used = action_counts.most_common(1)[0][0]
            failure = self.memory.detect_failure_pattern()
            reflection = f"Avg reward {avg_r:.3f}. Most used: {most_used}."
            if failure:
                reflection += f" DETECTED FAILURE: {failure} causing losses. Switching strategy."
            elif avg_r < 0.01:
                reflection += " Low returns. Diversify actions."
            else:
                reflection += " Current approach working."
            self.memory.add_reflection(reflection)
            return
        prompt = (
            f"You are {self.role}. Recent step history:\n"
            f"{json.dumps(recent, indent=2)}\n\n"
            "Write 1 sentence: what strategy worked or failed, and what to do next."
        )
        try:
            resp = await self.client.chat.completions.create(
                model=config.MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=80,
            )
            self.memory.add_reflection(resp.choices[0].message.content.strip())
        except Exception:
            pass


# ======================================================================
#  Board Parsing
# ======================================================================
def _parse_board_suggestions(board: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    combined = " ".join(board).lower()
    for act in ACTIONS:
        act_search = act.replace("_", " ")
        n = combined.count(act) + combined.count(act_search)
        if n > 0:
            counts[act] = n
    return counts


# ======================================================================
#  Policy
# ======================================================================
class Policy:
    """Orchestrates multi-agent coordination each step."""

    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    async def run_step(self, state: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, str]]:
        """Returns (final_action, board_messages, agent_votes_dict)."""
        roles = list(self.agents.keys())

        # 1. Discussion -- parallel
        discuss_tasks = [self.agents[r].discuss(state, []) for r in roles]
        board = list(await asyncio.gather(*discuss_tasks))

        # 2. Parse board for agreement signals
        board_suggestions = _parse_board_suggestions(board)

        # 3. Voting -- parallel, informed by board + agreement
        vote_tasks = [
            self.agents[r].vote(state, board, board_suggestions) for r in roles
        ]
        raw_votes = list(await asyncio.gather(*vote_tasks))

        agent_votes = {r: v for r, v in zip(roles, raw_votes)}

        # 4. Aggregate -- majority vote, alphabetical tie-break
        counts: Dict[str, int] = {}
        for v in raw_votes:
            if v in ACTIONS:
                counts[v] = counts.get(v, 0) + 1
        if not counts:
            counts[ACTIONS[0]] = 1
        final_action = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]

        return final_action, board, agent_votes


# ======================================================================
#  Factory
# ======================================================================
def make_agents() -> Dict[str, Agent]:
    return {role: Agent(role, defn) for role, defn in AGENT_DEFS.items()}
