def safe_score(traj):
    try:
        if not isinstance(traj, list) or len(traj) == 0:
            return 0.0
        rewards = []
        for step in traj:
            if isinstance(step, dict):
                r = step.get("reward", 0.0)
                if isinstance(r, (int, float)):
                    rewards.append(float(r))
        if not rewards:
            return 0.0
        score = sum(rewards) / len(rewards)
        return max(0.0, min(1.0, score))
    except:
        return 0.0

def grade_easy(traj):
    return safe_score(traj)

def grade_medium(traj):
    return safe_score(traj)

def grade_hard(traj):
    return safe_score(traj)
