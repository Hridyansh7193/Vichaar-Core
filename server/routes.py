"""API route handlers for the Vichaar-Core server."""
import copy
import uuid
import asyncio
import logging
from typing import Dict, Any, List

from fastapi import APIRouter

from core.env import Env
from agents import make_agents
from decision.policy import Policy
from evaluation.grader import compute_final_grade
from training.trajectory import TrajectoryCollector
from configs.env_config import ACTIONS
from configs.agent_config import AGENT_DEFS
from server.schemas import (
    Observation, ResetRequest, StepResponse,
    RunRequest, RunResponse, RunSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# -- shared instances (created once at import) --
env = Env()
agents = make_agents()
policy = Policy(agents)
collector = TrajectoryCollector()


@router.get("/")
async def root():
    return {"status": "active", "version": "2.0.0", "agents": list(AGENT_DEFS.keys())}


@router.get("/config")
async def get_config():
    return {
        "actions": ACTIONS,
        "agents": {k: v["desc"] for k, v in AGENT_DEFS.items()},
    }


@router.get("/state", response_model=Observation)
async def get_state():
    return env.state()


@router.post("/reset", response_model=Observation)
async def reset_env(req: ResetRequest):
    logger.info(f"Reset -> {req.task_id}")
    obs = env.reset(req.task_id)
    for a in agents.values():
        a.memory.clear()
    policy.safe_mode.reset()
    collector.reset()
    return obs


@router.post("/step", response_model=StepResponse)
async def step_env():
    state = env.state()
    if state["scenario"] == "Uninitialized":
        state = env.reset("medium")

    action, board, votes = await policy.run_step(state)
    next_obs, step_reward, done, info = env.step(action, messages=board, agent_votes=votes)
    rewards = info.get("rewards", {})

    for role, agent in agents.items():
        r = rewards.get(role, 0.0)
        agent.memory.add(state["metrics"], action, r, state["step_count"])
        agent.update_values(action, r)

    collector.log_step(state["step_count"], state, action, votes, rewards, next_obs, info)

    step_info = dict(info)
    step_info["decision"] = policy.decision_info

    return StepResponse(
        observation=next_obs,
        action=action,
        agent_votes=votes,
        reward=step_reward,
        done=done,
        info=step_info,
    )


@router.post("/run", response_model=RunResponse)
async def run_episode(req: RunRequest):
    logger.info(f"Run episode -> {req.task_id}")
    state = env.reset(req.task_id)
    for a in agents.values():
        a.memory.clear()
    policy.safe_mode.reset()
    collector.reset()

    episode_id = uuid.uuid4().hex[:8]
    max_steps = req.max_steps or env.max_steps

    history_log: List[Dict[str, Any]] = []
    totals = {r: 0.0 for r in agents}
    collab_count = 0

    for step in range(max_steps):
        action, board, votes = await policy.run_step(state)
        next_obs, step_reward, done, info = env.step(action, messages=board, agent_votes=votes)
        rewards = info.get("rewards", {})

        for role, agent in agents.items():
            r = rewards.get(role, 0.0)
            agent.memory.add(state["metrics"], action, r, step)
            agent.update_values(action, r)
            totals[role] += r

        if info.get("collaborated"):
            collab_count += 1

        di = policy.decision_info

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
            "decision_source": di.get("decision_source", "agents"),
            "ceo_override": di.get("ceo_override", False),
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
