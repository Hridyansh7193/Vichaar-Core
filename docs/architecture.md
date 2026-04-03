# System Architecture

## Overview

Vichaar-Core is a **4-layer strategic decision engine** where 5 specialized AI agents
collaborate under hierarchical control to navigate complex multi-stakeholder business scenarios.

## Decision Layers (Priority Order)

```
Layer 1: CEO (Hard Constraints)
   |  Overrides everything if critical thresholds breached
   |  cost > 0.85  -->  force reduce_cost
   |  legal_risk > 0.70  -->  force invest_in_safety
   |  env_impact > 0.70  -->  force green_innovation
   v
Layer 2: Safe Mode (Decline Detection)
   |  Monitors global score over 3-step window
   |  If score declining > 0.15  -->  ban risky actions
   |  Banned: launch_fast, outsource_tasks, increase_production
   v
Layer 3: Coordinator (Lookahead Planner)
   |  Simulates all 12 actions via 1-step lookahead
   |  Computes GLOBAL_SCORE for each simulated outcome
   |  Returns ranked list of actions (best to worst)
   v
Layer 4: Agent Voting (Distributed Intelligence)
   |  5 agents discuss on shared board
   |  Each agent votes using:
   |    - State-aware urgency scoring
   |    - Learned Q-values (updated from rewards)
   |    - Phase-based strategy (explore/optimize/correct/plan)
   |    - Memory-driven failure avoidance
   |    - Epsilon-greedy + softmax exploration
   v
Blending: Coordinator ranking + Agent votes = Final Action
   Coordinator top pick gets +2 bonus
   Coordinator top-3 get +1 bonus
   Highest blended score wins
```

## Module Map

| Module             | Responsibility                              |
|--------------------|---------------------------------------------|
| `configs/`         | All tunables (actions, thresholds, weights)  |
| `core/env.py`      | RL environment (reset/step/state)            |
| `core/reward.py`   | Per-agent delta-based rewards                |
| `agents/base.py`   | Agent class + MemoryStream                  |
| `decision/ceo.py`  | Hard constraint enforcement                 |
| `decision/safemode.py` | Score decline detection                  |
| `decision/coordinator.py` | 1-step lookahead planning            |
| `decision/policy.py` | 4-layer orchestration                      |
| `evaluation/`      | Final grading (0-1 scale)                   |
| `tasks/`           | 5 scenario definitions                      |
| `training/`        | Episode training + trajectory logging       |
| `api/`             | FastAPI server                              |
