"""
Inference -- run all scenarios with strategic decision engine visibility.
Shows: CEO overrides, Coordinator lookahead, Safe Mode, agent votes, learning.
"""

import os
from dotenv import load_dotenv

# Ensure environment is loaded strictly before anything else
load_dotenv()
import sys
import asyncio
import logging
import traceback
from collections import Counter
from core.env import Env
from agents import make_agents
from decision.policy import Policy
from decision.coordinator import compute_global_score
from evaluation.grader import compute_final_grade
from configs.agent_config import PHASE_STRATEGY

logger = logging.getLogger(__name__)

REFLECTION_EVERY = 5


async def run_episode(env: Env, policy: Policy, task_id: str) -> float:
    """Run a single episode and return the final grade."""
    obs = env.reset(task_id)
    policy.safe_mode.reset()

    g_score = compute_global_score(obs["metrics"])
    print(f"\n{'='*90}")
    print(f"  TASK: {task_id.upper()}  |  Max Steps: {env.max_steps}  |  Initial G-Score: {g_score:+.3f}")
    print(f"  {obs['scenario']}")
    print(f"{'='*90}")

    agent_totals = {r: 0.0 for r in policy.agents}
    actions_used = []
    collab_steps = 0
    ceo_steps = 0

    for step in range(env.max_steps):
        phase = obs.get("phase", "execution")
        phase_label = PHASE_STRATEGY.get(phase, {}).get("label", "?")

        try:
            action, board, votes = await policy.run_step(obs)
        except Exception as exc:
            logger.error(f"Policy error at step {step}: {exc}")
            action = "invest_in_safety"  # safe fallback
            board, votes = [], {r: action for r in policy.agents}

        actions_used.append(action)
        di = policy.decision_info

        next_obs, step_reward, done, info = env.step(
            action, messages=board, agent_votes=votes
        )
        rewards = info.get("rewards", {})

        for role, agent in policy.agents.items():
            r = rewards.get(role, 0.0)
            agent.memory.add(obs["metrics"], action, r, step)
            agent.update_values(action, r)
            agent_totals[role] += r

        if (step + 1) % REFLECTION_EVERY == 0:
            await asyncio.gather(
                *[a.reflect(obs) for a in policy.agents.values()]
            )

        collaborated = info.get("collaborated", False)
        if collaborated:
            collab_steps += 1
        if di.get("ceo_reason"):
            ceo_steps += 1

        collab_flag = "Y" if collaborated else "N"
        g_now = compute_global_score(next_obs["metrics"])
        g_delta = g_now - compute_global_score(obs["metrics"])

        deltas = {k: round(next_obs["metrics"][k] - obs["metrics"][k], 3) for k in obs["metrics"]}
        src = di.get("decision_source", "agents")

        print(
            f"  Step {step+1:>2} [{phase_label:>7}] | "
            f"{action:<22} | Co:{collab_flag} | "
            f"G={g_now:+.3f} (d{g_delta:+.3f}) | "
            f"Src: {src}"
        )
        print(
            f"         dP={deltas.get('expected_profit',0):+.3f} "
            f"dR={deltas.get('legal_risk',0):+.3f} "
            f"dE={deltas.get('env_impact',0):+.3f} "
            f"dS={deltas.get('public_sentiment',0):+.3f} "
            f"dC={deltas.get('cost',0):+.3f}"
        )

        if di.get("ceo_reason"):
            print(f"         !! {di['ceo_reason']}")
        else:
            coord_reason = di.get("coordinator_reason", "")
            if coord_reason:
                print(f"         Coord: {coord_reason}")
            vote_str = " ".join(f"{r[0].upper()}:{v[:10]}" for r, v in votes.items())
            print(f"         Votes: {vote_str}")

        if (step + 1) % REFLECTION_EVERY == 0:
            print(f"    --- REFLECTION (step {step+1}) | G-Score: {g_now:+.3f} ---")
            for role, agent in policy.agents.items():
                top3 = agent.top_values(3)
                q_str = ", ".join(f"{a}={v:+.3f}" for a, v in top3)
                fail = agent.memory.detect_failure_pattern()
                print(f"      [{role:>6}] Q: {q_str}")
                if fail:
                    print(f"               !! FAILURE: {fail}")
            if policy.safe_mode.active:
                print(f"      >> SAFE MODE ACTIVE -- risky actions banned")
            print(f"    ---")

        obs = next_obs
        if done:
            break

    grade = compute_final_grade(obs, task_id)
    g_final = compute_global_score(obs["metrics"])
    unique_actions = len(set(actions_used))
    action_counts = Counter(actions_used)
    most_common_3 = action_counts.most_common(3)

    print(f"{'~'*90}")
    print(f"  EPISODE SUMMARY -- {task_id.upper()}")
    print(f"  Grade: {grade:.3f}  |  G-Score: {g_final:+.3f}  |  Steps: {len(actions_used)}")
    print(f"  CEO Overrides: {ceo_steps}  |  Collabs: {collab_steps}  |  Safe Mode: {'YES' if policy.safe_mode.active else 'no'}")
    print(f"  Unique Actions: {unique_actions}/12  |  Top 3: {', '.join(f'{a}({c})' for a,c in most_common_3)}")
    print(f"  Agent Rewards: { {k: round(v, 3) for k, v in agent_totals.items()} }")
    print(f"  Final Metrics: { {k: round(v, 3) for k, v in obs['metrics'].items()} }")
    print(f"{'='*90}")
    return grade


async def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("\n" + "="*90)
    print("  Vichaar-Core -- Strategic Multi-Agent RL (Coordinator + CEO + Safe Mode)")
    print("="*90)

    from configs.api_config import OPENAI_API_BASE, MODEL_NAME, OPENAI_API_KEY
    print(f"  Configuration Loaded:")
    print(f"  - API_BASE_URL: {OPENAI_API_BASE}")
    print(f"  - MODEL_NAME:   {MODEL_NAME}")
    print(f"  - API Key:      {'Yes (set)' if OPENAI_API_KEY else 'No (heuristic mode)'}")
    print("="*90)

    env = Env()
    agents = make_agents()
    policy = Policy(agents)

    tasks = ["easy", "medium", "hard", "adversarial", "chaotic"]
    results = {}

    for tid in tasks:
        # Reset agent state between tasks for fair evaluation
        for a in agents.values():
            a.memory.clear()
            a.action_values = {act: 0.0 for act in a.action_values}
        policy.safe_mode.reset()

        try:
            grade = await run_episode(env, policy, tid)
        except Exception as exc:
            logger.error(f"Episode '{tid}' failed: {exc}")
            traceback.print_exc()
            grade = 0.0

        results[tid] = grade

    print("\n" + "="*54)
    print("  FINAL EVALUATION SCORES")
    print("="*54)
    for tid, score in results.items():
        bar_filled = int(score * 20)
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"  {tid:<13} [{bar}] {score:.3f}")
    avg = sum(results.values()) / len(results)
    print("-"*54)
    print(f"  Mean Grade:  {avg:.3f}")
    print("="*54)


if __name__ == "__main__":
    asyncio.run(main())
