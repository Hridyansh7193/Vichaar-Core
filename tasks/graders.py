def safe_score(traj):
    """Base helper to extract average reward from trajectory."""
    try:
        if not isinstance(traj, list) or len(traj) == 0:
            return 0.0
        rewards = [float(step.get("reward", 0.0)) for step in traj if isinstance(step, dict)]
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)
    except:
        return 0.0

def grade_easy(traj):
    """Easy: Simple reward average."""
    return max(0.001, min(0.999, safe_score(traj)))

def grade_medium(traj):
    """Medium: Penalty if agent didn't reach 'done' state."""
    score = safe_score(traj)
    try:
        last_step = traj[-1]
        if not last_step.get("done", False):
            score *= 0.8  # 20% penalty for incomplete task
    except:
        pass
    return max(0.001, min(0.999, score))

def grade_hard(traj):
    """Hard: Extra weight on final profit and survival."""
    score = safe_score(traj)
    try:
        final_obs = traj[-1].get("observation", {})
        metrics = final_obs.get("metrics", {})
        profit = metrics.get("expected_profit", 0.0)
        
        # Bonus for finishing with profit > 0.6
        if profit > 0.6:
            score += 0.1
        
        # Penalty if agent was forced to terminate by max_steps instead of finishing
        if not traj[-1].get("done", False):
            score -= 0.1
    except:
        pass
    return max(0.001, min(0.999, score))

def grade_adversarial(traj):
    """Adversarial: Critical fail if legal_risk ever crossed 0.8."""
    score = safe_score(traj)
    try:
        for step in traj:
            obs = step.get("observation", {})
            risk = obs.get("metrics", {}).get("legal_risk", 0.0)
            if risk > 0.8:
                return 0.05  # Severe penalty for governance failure
    except:
        pass
    return max(0.001, min(0.999, score))

def grade_chaotic(traj):
    """Chaotic: Measures stability. High rewards + low variance."""
    score = safe_score(traj)
    try:
        rewards = [float(s.get("reward", 0.0)) for s in traj]
        if len(rewards) > 1:
            # Simple variance penalty (fake variance for demo)
            variance = max(rewards) - min(rewards)
            score -= (variance * 0.1)
    except:
        pass
    return max(0.001, min(0.999, score))
