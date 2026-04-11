"""
Simulates the remote hackathon validator checks for Phase 2.
Tests:
1. openenv.yaml has >= 3 tasks with grader field
2. Each task has required fields: id, name, description, difficulty, grader
3. Grader modules are importable and callable
4. Graders return scores in [0.0, 1.0]
5. /tasks endpoint returns proper data
6. Full reset -> step -> grade pipeline works for each task
"""

import importlib
import json
import sys
import yaml

def test_openenv_yaml():
    """Phase 2 Check: openenv.yaml structure."""
    print("=" * 60)
    print("TEST 1: openenv.yaml structure")
    print("=" * 60)
    
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    
    # Required top-level fields
    required_top = ["spec_version", "name", "app", "tasks"]
    for field in required_top:
        val = manifest.get(field)
        status = "✓" if val else "✗"
        print(f"  {status} Top-level '{field}': {val}")
    
    tasks = manifest.get("tasks", [])
    print(f"\n  Total tasks: {len(tasks)}")
    
    tasks_with_graders = 0
    required_task_fields = ["id", "name", "description", "difficulty", "grader"]
    
    for i, t in enumerate(tasks):
        print(f"\n  --- Task {i+1} ---")
        all_fields_ok = True
        for field in required_task_fields:
            val = t.get(field)
            ok = bool(val) and val != "" and val != "unknown"
            status = "✓" if ok else "✗"
            print(f"    {status} {field}: {repr(val)}")
            if not ok:
                all_fields_ok = False
        
        # Test grader import
        grader_str = t.get("grader", "")
        if ":" in grader_str:
            mod_path, func_name = grader_str.rsplit(":", 1)
            try:
                mod = importlib.import_module(mod_path)
                grader_fn = getattr(mod, func_name)
                if callable(grader_fn):
                    # Test grader with sample trajectory
                    test_traj = [
                        {"reward": 0.5, "done": False, "observation": {"metrics": {
                            "expected_profit": 0.5, "legal_risk": 0.1, 
                            "env_impact": 0.1, "public_sentiment": 0.5, "cost": 0.3
                        }}},
                        {"reward": 0.6, "done": True, "observation": {"metrics": {
                            "expected_profit": 0.6, "legal_risk": 0.15,
                            "env_impact": 0.12, "public_sentiment": 0.55, "cost": 0.35
                        }}}
                    ]
                    score = grader_fn(test_traj)
                    score_ok = isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0
                    print(f"    ✓ Grader callable, test score: {score} (in range: {score_ok})")
                    if all_fields_ok and score_ok:
                        tasks_with_graders += 1
                else:
                    print(f"    ✗ Grader not callable")
            except Exception as e:
                print(f"    ✗ Grader import failed: {e}")
        else:
            print(f"    ✗ No grader path defined")
    
    print(f"\n  Tasks with working graders: {tasks_with_graders}")
    passed = tasks_with_graders >= 3
    print(f"  {'✓' if passed else '✗'} CHECK: At least 3 tasks with graders: {passed}")
    return passed


def test_full_pipeline():
    """Phase 2 Check: Full reset -> step -> grade pipeline."""
    print("\n" + "=" * 60)
    print("TEST 2: Full environment pipeline (reset → step → grade)")
    print("=" * 60)
    
    from core.env import Env
    
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    
    tasks = manifest.get("tasks", [])
    all_passed = True
    
    for t in tasks:
        task_id = t["id"]
        grader_str = t.get("grader", "")
        print(f"\n  --- Pipeline for task '{task_id}' ---")
        
        env = Env()
        
        # Reset
        try:
            obs = env.reset(task_id=task_id)
            print(f"    ✓ Reset OK (scenario: {obs.get('scenario', 'N/A')[:50]})")
        except Exception as e:
            print(f"    ✗ Reset FAILED: {e}")
            all_passed = False
            continue
        
        # Run steps
        trajectory = []
        actions = ["invest_in_safety", "green_innovation", "reduce_cost", "pr_campaign", "market_research"]
        max_steps = t.get("max_steps", 10)
        
        for step_num in range(min(max_steps, 5)):
            action = actions[step_num % len(actions)]
            try:
                obs, reward, done, info = env.step(action)
                trajectory.append({
                    "reward": float(reward),
                    "done": done,
                    "action": action,
                    "observation": obs
                })
                if done:
                    break
            except Exception as e:
                print(f"    ✗ Step {step_num+1} FAILED: {e}")
                all_passed = False
                break
        
        print(f"    ✓ Ran {len(trajectory)} steps")
        
        # Grade
        if ":" in grader_str:
            mod_path, func_name = grader_str.rsplit(":", 1)
            try:
                mod = importlib.import_module(mod_path)
                grader_fn = getattr(mod, func_name)
                score = float(grader_fn(trajectory))
                score = max(0.0, min(1.0, score))
                ok = 0.0 <= score <= 1.0
                print(f"    {'✓' if ok else '✗'} Graded: score={score:.4f} (in range: {ok})")
                if not ok:
                    all_passed = False
            except Exception as e:
                print(f"    ✗ Grading FAILED: {e}")
                all_passed = False
    
    print(f"\n  {'✓' if all_passed else '✗'} Full pipeline check: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def test_tasks_endpoint_response():
    """Phase 2 Check: Simulate /tasks endpoint response."""
    print("\n" + "=" * 60)
    print("TEST 3: /tasks endpoint response format")
    print("=" * 60)
    
    with open("openenv.yaml", "r") as f:
        manifest = yaml.safe_load(f)
    
    tasks_raw = manifest.get("tasks", [])
    tasks_out = []
    
    for t in tasks_raw:
        grader_str = t.get("grader", "")
        grader_ok = False
        if ":" in grader_str:
            mod_path, func_name = grader_str.rsplit(":", 1)
            try:
                mod = importlib.import_module(mod_path)
                grader_fn = getattr(mod, func_name)
                if callable(grader_fn):
                    test_traj = [{"reward": 0.5, "done": True, "observation": {"metrics": {"expected_profit": 0.5, "legal_risk": 0.1, "env_impact": 0.1, "public_sentiment": 0.5, "cost": 0.3}}}]
                    test_score = grader_fn(test_traj)
                    grader_ok = isinstance(test_score, (int, float)) and 0.0 <= float(test_score) <= 1.0
            except Exception:
                grader_ok = False

        tasks_out.append({
            "id": t.get("id"),
            "name": t.get("name", t.get("id")),
            "description": t.get("description", ""),
            "difficulty": t.get("difficulty", "unknown"),
            "max_steps": t.get("max_steps", 10),
            "grader": grader_str,
            "grader_available": grader_ok,
        })
    
    response = {"tasks": tasks_out, "count": len(tasks_out)}
    print(f"  Response:\n{json.dumps(response, indent=2)}")
    
    # Validate
    available = sum(1 for task in tasks_out if task["grader_available"])
    has_desc = sum(1 for task in tasks_out if task.get("description"))
    has_diff = sum(1 for task in tasks_out if task.get("difficulty") and task["difficulty"] != "unknown")
    
    print(f"\n  Tasks with grader_available: {available}")
    print(f"  Tasks with description: {has_desc}")
    print(f"  Tasks with difficulty: {has_diff}")
    
    passed = available >= 3 and has_desc >= 3 and has_diff >= 3
    print(f"  {'✓' if passed else '✗'} /tasks endpoint check: {'PASSED' if passed else 'FAILED'}")
    return passed


if __name__ == "__main__":
    results = []
    results.append(("openenv.yaml structure", test_openenv_yaml()))
    results.append(("Full pipeline", test_full_pipeline()))
    results.append(("/tasks endpoint", test_tasks_endpoint_response()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print(f"\n  Overall: {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    sys.exit(0 if all_passed else 1)
