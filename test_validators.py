from fastapi.testclient import TestClient
from app import app
import json

client = TestClient(app)

print("=== /reset ===")
r_reset = client.post("/reset", json={"task_id": "hard"})
print(json.dumps(r_reset.json(), indent=2))

print("\n=== /step ===")
r_step = client.post("/step")
print(json.dumps(r_step.json(), indent=2))

print("\n=== /run ===")
r_run = client.post("/run", json={"task_id": "medium", "max_steps": 3})
print(json.dumps(r_run.json(), indent=2))
