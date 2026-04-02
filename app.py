"""
FastAPI Application — Research-Grade Multi-Agent RL Simulation API

Endpoints:
  GET  /          → health check
  POST /reset     → reset environment to a task
  POST /step      → one coordinated multi-agent step
  POST /run       → full episode with trajectory logging
  GET  /state     → current environment state
  GET  /config    → current env_config values
"""

import logging
import copy
import uuid
import asyncio
from typing import Dict, Any, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from env import Env
from multi_agent import make_agents, Policy
from trajectory import TrajectoryCollector
from grader import compute_final_grade
from env_config import ACTIONS, AGENT_DEFS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vichaar-Core — Research-Grade Multi-Agent RL API",
    version="2.0.0",
)

# ── global instances ──────────────────────────────────────────────────
env = Env()
agents = make_agents()
policy = Policy(agents)
collector = TrajectoryCollector()


# ══════════════════════════════════════════════════════════════════════
#  Pydantic Models
# ══════════════════════════════════════════════════════════════════════
class Metrics(BaseModel):
    expected_profit: float
    legal_risk: float
    env_impact: float
    public_sentiment: float
    cost: float

class Observation(BaseModel):
    scenario: str
    phase: str
    metrics: Metrics
    entities: Dict[str, Any]
    events: List[str]
    history: List[str]
    step_count: int
    agent_messages: List[str]
    metrics_trend: List[Dict[str, float]]

class ResetRequest(BaseModel):
    task_id: str = "medium"

class StepResponse(BaseModel):
    observation: Observation
    action: str
    agent_votes: Dict[str, str]
    rewards: Dict[str, float]
    done: bool
    info: Dict[str, Any]

class RunRequest(BaseModel):
    task_id: str = "medium"
    max_steps: Optional[int] = None

class RunSummary(BaseModel):
    total_steps: int
    final_grade: float
    total_agent_rewards: Dict[str, float]
    performance: str
    collaborated_steps: int

class RunResponse(BaseModel):
    history: List[Dict[str, Any]]
    final_state: Observation
    summary: RunSummary


# ══════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════
@app.get("/")
async def root():
    return {"status": "active", "version": "2.0.0", "agents": list(AGENT_DEFS.keys())}

@app.get("/config")
async def get_config():
    return {
        "actions": ACTIONS,
        "agents": {k: v["desc"] for k, v in AGENT_DEFS.items()},
    }

@app.get("/state", response_model=Observation)
async def get_state():
    return env.state()

@app.post("/reset", response_model=Observation)
async def reset_env(req: ResetRequest):
    logger.info(f"Reset → {req.task_id}")
    obs = env.reset(req.task_id)
    for a in agents.values():
        a.memory.clear()
    collector.reset()
    return obs

@app.post("/step", response_model=StepResponse)
async def step_env():
    state = env.state()
    if state["scenario"] == "Uninitialized":
        state = env.reset("medium")

    action, board, votes = await policy.run_step(state)
    next_obs, rewards, done, info = env.step(action, messages=board, agent_votes=votes)

    # memory + learning
    for role, agent in agents.items():
        r = rewards.get(role, 0.0)
        agent.memory.add(state["metrics"], action, r, state["step_count"])
        agent.update_values(action, r)

    collector.log_step(state["step_count"], state, action, votes, rewards, next_obs, info)

    return StepResponse(
        observation=next_obs,
        action=action,
        agent_votes=votes,
        rewards=rewards,
        done=done,
        info=info,
    )

@app.post("/run", response_model=RunResponse)
async def run_episode(req: RunRequest):
    logger.info(f"Run episode → {req.task_id}")
    state = env.reset(req.task_id)
    for a in agents.values():
        a.memory.clear()
    collector.reset()

    episode_id = uuid.uuid4().hex[:8]
    max_steps = req.max_steps or env.max_steps

    history_log: List[Dict[str, Any]] = []
    totals = {r: 0.0 for r in agents}
    collab_count = 0

    for step in range(max_steps):
        action, board, votes = await policy.run_step(state)
        next_obs, rewards, done, info = env.step(action, messages=board, agent_votes=votes)

        for role, agent in agents.items():
            r = rewards.get(role, 0.0)
            agent.memory.add(state["metrics"], action, r, step)
            agent.update_values(action, r)
            totals[role] += r

        if info.get("collaborated"):
            collab_count += 1

        # periodic reflection
        if (step + 1) % 5 == 0:
            await asyncio.gather(*[a.reflect(state) for a in agents.values()])

        history_log.append({
            "step": step + 1,
            "phase": state.get("phase", ""),
            "action": action,
            "agent_votes": votes,
            "rewards": rewards,
            "metrics": copy.deepcopy(next_obs["metrics"]),
            "events": info.get("events", []),
            "collaborated": info.get("collaborated", False),
        })
        collector.log_step(step, state, action, votes, rewards, next_obs, info)

        state = next_obs
        if done:
            break

    collector.save_episode(episode_id)
    grade = compute_final_grade(state, req.task_id)

    if grade >= 0.70:
        perf = "Excellent"
    elif grade >= 0.40:
        perf = "Average"
    else:
        perf = "Critical"

    return RunResponse(
        history=history_log,
        final_state=state,
        summary=RunSummary(
            total_steps=len(history_log),
            final_grade=grade,
            total_agent_rewards={k: round(v, 3) for k, v in totals.items()},
            performance=perf,
            collaborated_steps=collab_count,
        ),
    )
