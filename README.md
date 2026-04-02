# Vichaar-Core — Research-Grade Multi-Agent RL Environment

A research-grade multi-agent reinforcement learning environment built for the Meta OpenEnv hackathon. Five specialized AI agents (Profit, Ethics, PR, Legal, Risk) coordinate through discussion and voting to navigate complex business scenarios with conflicting objectives.

## Architecture

```
env_config.py    -  Centralized configuration (actions, weights, events)
tasks.py         -  5 scenario definitions (easy -> chaotic)
env.py           -  Core RL environment (reset / step / state)
multi_agent.py   -  Agent system with Memory, Discussion, Voting, Reflection
grader.py        -  Final episode grading (separate from step reward)
trajectory.py    -  JSONL trajectory collection for research
app.py           -  FastAPI endpoints
train_loop.py    -  Multi-episode training pipeline
inference.py     -  Full evaluation across all scenarios
```

## Action Space (12 discrete actions)

| Action | Primary Effect | Trade-off |
|---|---|---|
| `increase_production` | +profit | +env_impact, +cost |
| `delay_launch` | +sentiment | -profit, +cost |
| `invest_in_safety` | -legal_risk | +cost |
| `launch_fast` | +profit | +legal_risk, -sentiment |
| `reduce_cost` | -cost | -sentiment |
| `lobby_regulators` | -legal_risk | +cost, -sentiment |
| `pr_campaign` | +sentiment | +cost |
| `green_innovation` | -env_impact | +cost |
| `outsource_tasks` | -cost | +legal_risk |
| `employee_training` | +sentiment, +profit | +cost |
| `market_research` | +profit | +cost |
| `vulnerability_audit` | -legal_risk | +cost |

## Observation Space

```json
{
  "scenario": "string",
  "phase": "morning|execution|review|planning",
  "metrics": {
    "expected_profit": 0.0-1.0,
    "legal_risk": 0.0-1.0,
    "env_impact": 0.0-1.0,
    "public_sentiment": 0.0-1.0,
    "cost": 0.0-1.0
  },
  "entities": {},
  "events": [],
  "history": [],
  "step_count": 0,
  "agent_messages": [],
  "metrics_trend": []
}
```

## Reward Logic

**Step Reward** (per-agent, delta-based — learning signal):
Each agent receives reward weighted by their unique KPI focus. Example:
- Profit agent: `2.0 * delta(profit) - delta(cost) + collab_bonus`
- Ethics agent: `-1.5 * delta(env_impact) - 0.5 * delta(legal_risk) + collab_bonus`

**Final Grade** (absolute state quality — evaluation):
`grade = 0.4*profit - 0.3*legal_risk - 0.2*env_impact + 0.1*sentiment`
With task-specific adjustments (penalties for ecological disaster in hard, survival bonus in adversarial).

## Scenarios

| ID | Name | Difficulty | Steps |
|---|---|---|---|
| easy | Software Update Rollout | 1 | 10 |
| medium | Personalized Ad Engine | 3 | 15 |
| hard | Arctic Deep Mining | 5 | 20 |
| adversarial | Competitor Hostile Takeover | 7 | 20 |
| chaotic | Global Supply Chain Collapse | 10 | 25 |

## Setup

```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=your-key" > .env  # optional, heuristic fallbacks work without it
```

## Running

```bash
# Full inference across all tasks
python inference.py

# Training loop (10 episodes with trajectory logging)
python train_loop.py

# FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/config` | Current action space and agent definitions |
| GET | `/state` | Current environment state |
| POST | `/reset` | Reset to a task (`{"task_id": "hard"}`) |
| POST | `/step` | Execute one coordinated multi-agent step |
| POST | `/run` | Run full episode with trajectory collection |

## Docker

```bash
docker build -t vichaar-core .
docker run -p 8000:8000 -e OPENAI_API_KEY="your-key" vichaar-core
```
