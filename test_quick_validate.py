"""Quick validation test - simulates remote validator Phase 2 checks."""
import importlib
import yaml
import sys

with open("openenv.yaml", "r") as f:
    manifest = yaml.safe_load(f)

tasks = manifest.get("tasks", [])
print("Total tasks:", len(tasks))

valid_tasks = 0
for t in tasks:
    tid = t.get("id", "?")
    has_id = bool(tid)
    has_name = bool(t.get("name"))
    has_desc = bool(t.get("description"))
    has_diff = bool(t.get("difficulty")) and t.get("difficulty") != "unknown"
    has_grader = bool(t.get("grader"))

    grader_works = False
    score = -1
    grader_str = t.get("grader", "")
    if ":" in grader_str:
        mod_path, func_name = grader_str.rsplit(":", 1)
        try:
            mod = importlib.import_module(mod_path)
            fn = getattr(mod, func_name)
            test_traj = [{"reward": 0.5, "done": True, "observation": {"metrics": {
                "expected_profit": 0.5, "legal_risk": 0.1,
                "env_impact": 0.1, "public_sentiment": 0.5, "cost": 0.3
            }}}]
            score = fn(test_traj)
            grader_works = isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0
        except Exception as e:
            print("  Grader error for %s: %s" % (tid, e))

    all_ok = all([has_id, has_name, has_desc, has_diff, has_grader, grader_works])
    status = "PASS" if all_ok else "FAIL"
    print("  Task %s: id=%s name=%s desc=%s diff=%s grader=%s works=%s score=%.3f => %s" % (
        tid, has_id, has_name, has_desc, has_diff, has_grader, grader_works, score, status))
    if all_ok:
        valid_tasks += 1

print("")
print("Valid tasks with graders: %d/5" % valid_tasks)
passed = valid_tasks >= 3
print("Phase 2 check (>=3): %s" % ("PASS" if passed else "FAIL"))
sys.exit(0 if passed else 1)
