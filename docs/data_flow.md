# Data Flow

## Per-Step Execution

```
1. OBSERVE
   State = { scenario, phase, metrics(5), entities, history, events }

2. CEO CHECK (decision/ceo.py)
   if cost >= 0.85       --> OVERRIDE: force "reduce_cost"
   if legal_risk >= 0.70 --> OVERRIDE: force "invest_in_safety"
   if env_impact >= 0.70 --> OVERRIDE: force "green_innovation"
   --> If triggered, skip all other layers. Return forced action.

3. SAFE MODE CHECK (decision/safemode.py)
   Track global_score over last 3 steps
   if score declined > 0.15 --> ACTIVATE: ban risky actions
   Banned actions: launch_fast, outsource_tasks, increase_production

4. COORDINATOR LOOKAHEAD (decision/coordinator.py)
   For each of 12 actions:
     - Copy current metrics
     - Apply ACTION_EFFECTS
     - Clamp to [0, 1]
     - Compute GLOBAL_SCORE of simulated state
   Return: ranked list (best action first)

5. AGENT DISCUSSION (agents/base.py)
   Each of 5 agents posts a proposal to shared board
   Example: "[profit] cost is too high (0.80). I propose reduce_cost."

6. AGENT VOTING (agents/base.py)
   Each agent scores all 12 actions using:
     - Urgency scoring (how badly does each metric need fixing?)
     - Learned Q-values (alpha=0.3 temporal difference update)
     - Phase strategy (morning=explore, execution=optimize, etc.)
     - Repetition penalty (avoid repeating last 3 actions)
     - Board agreement bonus (+0.2 per mention)
     - Memory avoidance (skip actions with avg reward < -0.05)
     - Failure pattern detection (hard penalty -0.5)
   Select via epsilon-greedy + softmax sampling

7. BLEND (decision/policy.py)
   For each action:
     score = agent_vote_count
     if action == coordinator_top_pick: score += 2.0
     if action in coordinator_top_3:   score += 1.0
     if safe_mode AND action is risky:  score = -999
   Final action = argmax(blend_scores)

8. EXECUTE (core/env.py)
   Apply ACTION_EFFECTS to metrics
   Clamp all metrics to [0, 1]
   Fire stochastic events (seed-controlled)
   Compute per-agent rewards from metric deltas
   Advance phase (morning -> execution -> review -> planning)

9. LEARN
   Each agent:
     - Stores (state, action, reward) in MemoryStream
     - Updates Q-values: Q[a] += alpha * (reward - Q[a])
     - Every 5 steps: reflect on performance

10. REPEAT until max_steps reached
```

## Episode Flow

```
reset(task_id) --> 10-25 steps --> compute_final_grade() --> 0.0 to 1.0
```

## Reward vs Grade

| Signal           | Scope     | Purpose                    | Lives in          |
|------------------|-----------|----------------------------|--------------------|
| Per-agent reward | Per step  | Guide agent learning       | core/reward.py     |
| Global score     | Per step  | Guide Coordinator planning | evaluation/scoring |
| Final grade      | Per ep    | Judge submission quality    | evaluation/grader  |
