"""
Training Loop — run episodes, collect trajectories, update agent memory.

Demonstrates long-term learning through memory accumulation and periodic
reflection.  When an LLM is available, reflections become genuine strategy
updates; without one, heuristic reflections still adapt behavior.
"""

import asyncio
import logging
from typing import List

from env import Env
from multi_agent import make_agents, Policy
from grader import compute_final_grade
from trajectory import TrajectoryCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s")
logger = logging.getLogger(__name__)

REFLECTION_EVERY = 5


async def train_episode(
    env: Env,
    policy: Policy,
    collector: TrajectoryCollector,
    task_id: str,
    episode_idx: int,
) -> float:
    obs = env.reset(task_id)
    collector.reset()

    agent_totals = {r: 0.0 for r in policy.agents}

    for step in range(env.max_steps):
        action, board, votes = await policy.run_step(obs)
        next_obs, rewards, done, info = env.step(
            action, messages=board, agent_votes=votes
        )

        # memory + learning + trajectory
        for role, agent in policy.agents.items():
            r = rewards.get(role, 0.0)
            agent.memory.add(obs["metrics"], action, r, step)
            agent.update_values(action, r)
            agent_totals[role] += r

        collector.log_step(step, obs, action, votes, rewards, next_obs, info)

        # periodic reflection (long-term learning)
        if (step + 1) % REFLECTION_EVERY == 0:
            await asyncio.gather(*[a.reflect(obs) for a in policy.agents.values()])

        obs = next_obs
        if done:
            break

    saved = collector.save_episode(f"{task_id}_ep{episode_idx}")
    grade = compute_final_grade(obs, task_id)
    logger.info(
        f"Episode {episode_idx} [{task_id}] → grade={grade:.3f}  "
        f"steps={saved}  rewards={{{', '.join(f'{k}:{v:.3f}' for k,v in agent_totals.items())}}}"
    )
    return grade


async def train_loop(
    episodes: int = 10,
    tasks: List[str] | None = None,
):
    if tasks is None:
        tasks = ["easy", "medium", "hard", "adversarial", "chaotic"]

    env = Env()
    agents = make_agents()
    policy = Policy(agents)
    collector = TrajectoryCollector(output_dir="training_trajectories")

    logger.info(f"Starting training: {episodes} episodes across {tasks}")

    grades = []
    for i in range(1, episodes + 1):
        tid = tasks[(i - 1) % len(tasks)]
        grade = await train_episode(env, policy, collector, tid, i)
        grades.append(grade)

    avg = sum(grades) / len(grades) if grades else 0.0
    logger.info(f"Training complete.  Mean grade over {len(grades)} episodes: {avg:.3f}")
    logger.info("Agent memory streams contain accumulated experience for future episodes.")


if __name__ == "__main__":
    asyncio.run(train_loop(episodes=10))
