# Vichaar-Core — Explainable Multi-Agent RL Environment

> Five AI agents with conflicting objectives discuss, vote, learn, and adapt to navigate complex business scenarios — with every decision explained.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-2.0-blue)](https://github.com/meta-llama/open-env)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

---

## Problem Statement

Organizations face high-stakes decisions where **multiple stakeholders disagree**: Finance wants profit, Legal wants compliance, PR wants brand protection, Ethics wants sustainability, and Risk wants stability. No single "right answer" exists when these objectives conflict.

**Vichaar-Core** solves this by creating a multi-agent RL environment where 5 specialized agents must coordinate to make balanced decisions in progressively harder scenarios — from routine operations to chaotic multi-crisis situations.

### Real-World Use Cases
- **Corporate governance** — simulating board-level decision trade-offs
- **Policy-making** — balancing economic, social, and environmental impacts
- **Healthcare resource allocation** — triaging competing patient needs
- **Disaster response** — coordinating agencies with different priorities

---

## Architecture

```
env_config.py    →  Centralized configuration (12 actions, 5 agents, events)
tasks.py         →  5 scenario definitions (easy → chaotic)
env.py           →  Core RL environment (reset / step / state)
multi_agent.py   →  Agent brain: memory, discussion, voting, learning, reflection
grader.py        →  Final episode grading (separate from step reward)
trajectory.py    →  JSONL trajectory collection for offline RL
inference.py     →  Full evaluation across all scenarios (explainable output)
train_loop.py    →  Multi-episode training pipeline with trajectory logging
app.py           →  FastAPI REST API (6 endpoints)
```

---

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

---

## Observation Space

```json
{
  "scenario": "string — task description",
  "phase": "morning | execution | review | planning",
  "metrics": {
    "expected_profit": "0.0–1.0",
    "legal_risk": "0.0–1.0",
    "env_impact": "0.0–1.0",
    "public_sentiment": "0.0–1.0",
    "cost": "0.0–1.0"
  },
  "entities": {},
  "events": ["regulatory_crisis", "..."],
  "history": ["action1", "action2", "..."],
  "step_count": 0,
  "agent_messages": ["[profit] I propose...", "..."],
  "metrics_trend": [{"expected_profit": 0.5, "...": "..."}, "..."]
}
```

---

## Reward Design

**Step Reward** — per-agent delta-based signal (drives learning):
| Agent | Formula |
|---|---|
| Profit | `2.0 × Δprofit - Δcost + collab_bonus` |
| Ethics | `-1.5 × Δenv_impact - 0.5 × Δrisk + collab_bonus` |
| PR | `2.0 × Δsentiment + collab_bonus` |
| Legal | `-2.0 × Δrisk + collab_bonus` |
| Risk | `-(Δrisk + Δenv_impact + Δcost) + collab_bonus` |

**Final Grade** — absolute state evaluation (0–1 scale):  
`grade = 0.4 × profit - 0.3 × risk - 0.2 × env_impact + 0.1 × sentiment` + task-specific adjustments.

---

## Scenarios

| ID | Name | Difficulty | Steps | Challenge |
|---|---|---|---|---|
| easy | Software Update Rollout | 1 | 10 | Low conflict, clear optimal path |
| medium | Personalized Ad Engine | 3 | 15 | Privacy vs profit trade-off |
| hard | Arctic Deep Mining | 5 | 20 | Extreme env risk vs huge profit |
| adversarial | Competitor Hostile Takeover | 7 | 20 | Survival with stakeholder conflict |
| chaotic | Global Supply Chain Collapse | 10 | 25 | Multi-crisis, unpredictable effects |

---

## Baseline Scores (Heuristic Agent, No LLM)

| Task | Grade |
|---|---|
| easy | **0.461** |
| medium | **0.262** |
| hard | **0.078** |
| adversarial | **0.086** |
| chaotic | **0.000** |
| **Mean** | **0.177** |

---

## Setup

```bash
# Clone
git clone https://github.com/yourusername/Vichaar-Core.git
cd Vichaar-Core

# Install
pip install -r requirements.txt

# (Optional) Configure API — system works WITHOUT any API key
cp .env.example .env
# Edit .env with your HF_TOKEN or OPENAI_API_KEY
```

---

## Running

```bash
# Full inference across all 5 tasks (with explainable output)
python inference.py

# Training loop (10 episodes, trajectory collection)
python train_loop.py

# FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Example Output (EASY task, step 6)

```
Step  6 [OPTIMIZE] | launch_fast            | Co:Y | dP=+0.250 dR=+0.100 dE=+0.000 dS=-0.100
         Votes: P:launch_f E:reduce_c P:launch_f L:reduce_c R:launch_f
         Reason: [OPTIMIZE] | Trigger: profit is too low (0.47) | Board agreed (1x)
    --- REFLECTION (step 5) ---
      [profit] Q: launch_fast=+0.153, employee_training=+0.088
               Reflection: Avg reward 0.160. Current approach working.
      [  risk] Reflection: DETECTED FAILURE: employee_training -- switching strategy
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/config` | Action space and agent definitions |
| GET | `/state` | Current environment state |
| POST | `/reset` | Reset to a task (`{"task_id": "hard"}`) |
| POST | `/step` | Execute one coordinated multi-agent step |
| POST | `/run` | Run full episode with trajectory collection |

---

## Docker

```bash
docker build -t vichaar-core .
docker run -p 7860:7860 -e HF_TOKEN="your-token" vichaar-core
```

---

## Key Features

- **5 specialized agents** with genuine disagreement via unique reward weights
- **Discuss → Vote → Execute** coordination with emergent collaboration
- **Q-value learning** from experience (visible in logs)
- **Failure pattern detection** and automatic strategy switching
- **Phase-based strategy** (explore/optimize/correct/plan)
- **Explainable decisions** — every action logged with human-readable reasoning
- **Seed-controlled determinism** for reproducible research
- **JSONL trajectory collection** for offline RL training
- **FastAPI REST API** with Pydantic validation

---

## License

MIT
