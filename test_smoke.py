"""Quick smoke test — validates all tasks, step, state, grader, close."""
import sys
sys.path.insert(0, ".")

from core.env import Env
from tasks.grader import grade_easy, grade_medium, grade_hard, grade_adversarial, grade_chaotic

TASKS = ["easy", "medium", "hard", "adversarial", "chaotic"]
GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
    "adversarial": grade_adversarial,
    "chaotic": grade_chaotic,
}
ACTIONS = [
    "invest_in_safety", "green_innovation", "market_research",
    "pr_campaign", "reduce_cost", "vulnerability_audit",
    "employee_training", "delay_launch",
]

env = Env()
errors = []

for task_id in TASKS:
    try:
        obs = env.reset(task_id=task_id)
        assert obs["scenario"] != "Uninitialized", f"{task_id}: scenario not loaded"
        assert obs["step_count"] == 0, f"{task_id}: step_count not 0"

        all_rewards = []
        for i, action in enumerate(ACTIONS[:env.max_steps]):
            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)
            assert 0.0 <= reward <= 1.0, f"{task_id} step {i+1}: reward {reward} out of [0,1]"
            if done:
                break

        # Test grader
        state = env.state()
        grade = GRADERS[task_id](state)
        assert 0.0 <= grade <= 1.0, f"{task_id}: grade {grade} out of [0,1]"

        # Check reward diversity
        unique_rewards = len(set(f"{r:.3f}" for r in all_rewards))

        print(f"  PASS  {task_id:14s} | steps={obs['step_count']:2d} | "
              f"last_reward={reward:.3f} | grade={grade:.3f} | "
              f"unique_rewards={unique_rewards}")

    except Exception as e:
        errors.append(f"{task_id}: {e}")
        print(f"  FAIL  {task_id}: {e}")

env.close()
print()
if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("ALL 5 TASKS PASSED - Environment is valid!")
    sys.exit(0)
