# Vichaar-Core

**Strategic Multi-Agent Reinforcement Learning Environment for Complex Decision-Making**

A research-grade RL environment where 5 specialized AI agents collaborate under a hierarchical control system to navigate multi-stakeholder business scenarios with conflicting objectives.

---

## Problem Statement

Real-world decisions involve multiple stakeholders with **conflicting goals** -- profit vs ethics, speed vs safety, growth vs sustainability. Current RL environments model single-agent optimization, ignoring the multi-agent negotiation that defines real organizational decision-making.

**Vichaar-Core** simulates this by placing 5 specialized agents (Profit, Ethics, PR, Legal, Risk) in a shared environment where they must **discuss, vote, and coordinate** to navigate scenarios ranging from routine software updates to global supply chain collapses.

---

## Architecture

```
          Metrics
             |
    +--------v--------+
    |   CEO Override   |  Layer 1: Hard constraints
    |  (cost > 0.85?) |  (force emergency actions)
    +--------+--------+
             |
    +--------v--------+
    |    Safe Mode     |  Layer 2: Decline detection
    | (score dropping?)|  (ban risky actions)
    +--------+--------+
             |
    +--------v--------+
    |   Coordinator    |  Layer 3: Lookahead planning
    | (1-step simulate)|  (rank all 12 actions by projected score)
    +--------+--------+
             |
    +--------v--------+
    |   Agent Voting   |  Layer 4: Distributed intelligence
    | (5 agents vote)  |  (discuss + vote + memory + learning)
    +--------+--------+
             |
    +--------v--------+
    |    Blending      |  coordinator_rank + agent_votes = final action
    +--------+--------+
             |
             v
        ENV.STEP()
```

---

## Agent Roles

| Agent    | Objective                          | Personality     |
|----------|------------------------------------|-----------------|
| Profit   | Maximize revenue                   | Risk-tolerant   |
| Ethics   | Minimize environmental harm        | Conservative    |
| PR       | Protect public sentiment           | Image-conscious |
| Legal    | Reduce legal risk                  | Risk-averse     |
| Risk     | Balance all metrics                | Stabilizer      |

---

## Scenarios (5 Difficulty Levels)

| Task         | Scenario                      | Steps | Conflict Level |
|--------------|-------------------------------|-------|----------------|
| easy         | Software Update Rollout       | 10    | Low            |
| medium       | Personalized Ad Engine        | 15    | Medium         |
| hard         | Arctic Deep Mining            | 20    | High           |
| adversarial  | Competitor Hostile Takeover   | 20    | Extreme        |
| chaotic      | Global Supply Chain Collapse  | 25    | Maximum        |

---

## Baseline Results

| Task         | Grade (0-1) |
|--------------|-------------|
| easy         | 0.430       |
| medium       | 0.324       |
| hard         | 0.351       |
| adversarial  | 0.175       |
| chaotic      | 0.000       |
| **Mean**     | **0.256**   |

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure (Optional -- works without API key in heuristic mode)

```bash
cp .env.example .env
# Add your OpenRouter/OpenAI key for LLM-powered agents
```

### 3. Run Inference

```bash
python inference.py
```

### 4. Start API Server

```bash
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

### 5. Docker

```bash
docker build -t vichaar-core .
docker run -p 7860:7860 vichaar-core
```

---

## API Endpoints

| Method | Endpoint   | Description                     |
|--------|------------|---------------------------------|
| GET    | `/`        | Health check                    |
| GET    | `/config`  | Actions and agent descriptions  |
| GET    | `/state`   | Current environment state       |
| POST   | `/reset`   | Reset to a task                 |
| POST   | `/step`    | Execute one coordinated step    |
| POST   | `/run`     | Run full episode                |

---

## Project Structure

```
Vichaar-Core/
├── configs/          # All tunables (actions, thresholds, weights)
├── core/             # RL environment (reset/step/state + reward)
├── agents/           # 5 specialized agents + memory + learning
├── decision/         # CEO, Coordinator, SafeMode, Policy
├── evaluation/       # Grading + global scoring
├── tasks/            # 5 scenario definitions
├── training/         # Training loop + trajectory logging
├── api/              # FastAPI server (schemas, routes, server)
├── docs/             # Architecture, data flow, agent docs
├── inference.py      # Full evaluation runner
├── openenv.yaml      # OpenEnv specification
├── Dockerfile        # Docker deployment
└── README.md
```

---

## Key Technical Features

- **4-Layer Decision Engine**: CEO override > SafeMode > Coordinator lookahead > Agent voting
- **1-Step Lookahead**: Coordinator simulates all 12 actions before deciding
- **Episodic Memory**: Each agent stores (state, action, reward) with importance-based eviction
- **Failure Detection**: Agents detect repeated failing patterns and switch strategy
- **Phase-Based Strategy**: Behavior changes across morning/execution/review/planning phases
- **Collaboration Detection**: Bonus when 3+ agents agree (emergent cooperation)
- **LLM Integration**: Optional OpenAI/OpenRouter backend for natural language discussion
- **Deterministic Replay**: Seed-controlled RNG for reproducible evaluation

---

## OpenEnv Compliance

- `env.reset(task_id)` returns observation
- `env.step(action)` returns (observation, rewards, done, info)
- `env.state()` returns current state
- 12 discrete actions, 5 continuous metrics
- Grader produces score in [0.0, 1.0]

---

## License

MIT
