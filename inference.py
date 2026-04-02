"""
Inference -- run all scenarios with full explainability and learning visibility.
"""

import asyncio
from collections import Counter
from env import Env
from multi_agent import make_agents, Policy, PHASE_STRATEGY
from grader import compute_final_grade


REFLECTION_EVERY = 5


async def run_episode(env: Env, policy: Policy, task_id: str) -> float:
    obs = env.reset(task_id)

    print(f"\n{'='*80}")
    print(f"  TASK: {task_id.upper()}  |  Max Steps: {env.max_steps}")
    print(f"  {obs['scenario']}")
    print(f"{'='*80}")

    agent_totals = {r: 0.0 for r in policy.agents}
    actions_used = []
    collab_steps = 0

    for step in range(env.max_steps):
        phase = obs.get("phase", "execution")
        phase_label = PHASE_STRATEGY.get(phase, {}).get("label", "?")

        # -- coordination --
        action, board, votes = await policy.run_step(obs)
        actions_used.append(action)

        # -- step --
        next_obs, rewards, done, info = env.step(
            action, messages=board, agent_votes=votes
        )

        # -- memory + learning --
        for role, agent in policy.agents.items():
            r = rewards.get(role, 0.0)
            agent.memory.add(obs["metrics"], action, r, step)
            agent.update_values(action, r)
            agent_totals[role] += r

        # -- periodic reflection --
        if (step + 1) % REFLECTION_EVERY == 0:
            await asyncio.gather(
                *[a.reflect(obs) for a in policy.agents.values()]
            )

        # -- track collaboration --
        collaborated = info.get("collaborated", False)
        if collaborated:
            collab_steps += 1
        collab_flag = "Y" if collaborated else "N"

        # -- metric deltas --
        deltas = {k: round(next_obs["metrics"][k] - obs["metrics"][k], 3) for k in obs["metrics"]}

        # -- vote breakdown --
        vote_str = " ".join(f"{r[0].upper()}:{v[:8]}" for r, v in votes.items())

        # -- decision reason (from winning agent or majority) --
        # Find agents who voted for the winning action
        winning_agents = [r for r, v in votes.items() if v == action]
        reason = ""
        if winning_agents:
            reason = policy.agents[winning_agents[0]].last_reason

        # -- print step --
        print(
            f"  Step {step+1:>2} [{phase_label:>7}] | "
            f"{action:<22} | Co:{collab_flag} | "
            f"dP={deltas.get('expected_profit',0):+.3f} "
            f"dR={deltas.get('legal_risk',0):+.3f} "
            f"dE={deltas.get('env_impact',0):+.3f} "
            f"dS={deltas.get('public_sentiment',0):+.3f}"
        )
        print(
            f"         Votes: {vote_str}"
        )
        if reason:
            print(f"         Reason: {reason}")

        # -- show learning progress every 5 steps --
        if (step + 1) % REFLECTION_EVERY == 0:
            print(f"    --- REFLECTION (step {step+1}) ---")
            for role, agent in policy.agents.items():
                top3 = agent.top_values(3)
                q_str = ", ".join(f"{a}={v:+.3f}" for a, v in top3)
                refl = agent.memory.latest_reflection()
                fail = agent.memory.detect_failure_pattern()
                print(f"      [{role:>6}] Q: {q_str}")
                print(f"               Reflection: {refl[:80]}")
                if fail:
                    print(f"               !! FAILURE DETECTED: {fail} -- switching strategy")
            print(f"    ---")

        obs = next_obs
        if done:
            break

    # -- episode summary --
    grade = compute_final_grade(obs, task_id)
    unique_actions = len(set(actions_used))
    action_counts = Counter(actions_used)
    most_common_3 = action_counts.most_common(3)

    print(f"{'~'*80}")
    print(f"  EPISODE SUMMARY")
    print(f"  Grade: {grade:.3f}  |  Steps: {len(actions_used)}  |  Collabs: {collab_steps}")
    print(f"  Unique Actions: {unique_actions}/12  |  Top 3: {', '.join(f'{a}({c})' for a,c in most_common_3)}")
    print(f"  Agent Rewards: { {k: round(v, 3) for k, v in agent_totals.items()} }")
    print(f"  Final Metrics: { {k: round(v, 3) for k, v in obs['metrics'].items()} }")
    print(f"{'='*80}")
    return grade


async def main():
    print("\n" + "="*80)
    print("  Vichaar-Core -- Multi-Agent RL Research Inference (Explainable Mode)")
    print("="*80)

    env = Env()
    agents = make_agents()
    policy = Policy(agents)

    tasks = ["easy", "medium", "hard", "adversarial", "chaotic"]
    results = {}

    for tid in tasks:
        for a in agents.values():
            a.memory.clear()
            a.action_values = {act: 0.0 for act in a.action_values}
        grade = await run_episode(env, policy, tid)
        results[tid] = grade

    print("\n" + "="*50)
    print("  FINAL EVALUATION SCORES")
    print("="*50)
    for tid, score in results.items():
        bar_filled = int(score * 20)
        bar = "#" * bar_filled + "." * (20 - bar_filled)
        print(f"  {tid:<13} [{bar}] {score:.3f}")
    avg = sum(results.values()) / len(results)
    print("-"*50)
    print(f"  Mean Grade:  {avg:.3f}")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(main())
