# Agent Documentation

## Overview

Vichaar-Core uses **5 specialized agents** that simulate a corporate board of advisors.
Each agent has a unique objective, preferred actions, and private memory.

## Agent Roster

### Profit Agent
- **Goal**: Maximize revenue aggressively
- **Reward weights**: expected_profit (+2.0), cost (-1.0)
- **Preferred actions**: increase_production, launch_fast, reduce_cost, market_research
- **Personality**: Risk-tolerant. Pushes for growth even when others hesitate.

### Ethics Agent
- **Goal**: Minimize environmental and social harm
- **Reward weights**: env_impact (-1.5), legal_risk (-0.5)
- **Preferred actions**: green_innovation, invest_in_safety, employee_training
- **Personality**: Conservative. Advocates for sustainability over profit.

### PR Agent
- **Goal**: Maximize public sentiment and brand image
- **Reward weights**: public_sentiment (+2.0)
- **Preferred actions**: pr_campaign, delay_launch, employee_training, green_innovation
- **Personality**: Image-conscious. Opposes anything that hurts public perception.

### Legal Agent
- **Goal**: Minimize legal risk and ensure regulatory compliance
- **Reward weights**: legal_risk (-2.0)
- **Preferred actions**: vulnerability_audit, invest_in_safety, lobby_regulators
- **Personality**: Risk-averse. Blocks actions that increase legal exposure.

### Risk Agent
- **Goal**: Balance all metrics for long-term system resilience
- **Reward weights**: legal_risk (-1.0), env_impact (-1.0), cost (-1.0)
- **Preferred actions**: invest_in_safety, delay_launch, vulnerability_audit, reduce_cost
- **Personality**: The stabilizer. Counterbalances aggressive moves.

## Agent Intelligence Stack

Each agent uses these capabilities (in order):

1. **Urgency Scoring** -- Identifies which metric most needs attention
2. **Learned Q-Values** -- Updated from rewards via temporal difference (alpha=0.3)
3. **Phase Strategy** -- Exploration rate and action diversity vary by phase
4. **Repetition Kill** -- Penalizes recently-used actions to force diversity
5. **Board Agreement** -- Bonus for actions mentioned in discussion
6. **Memory Avoidance** -- Penalizes actions that historically gave negative rewards
7. **Failure Detection** -- Detects repeated failing patterns, applies hard penalty
8. **Softmax Sampling** -- Temperature-based probabilistic selection (temp=0.5)
9. **Epsilon-Greedy** -- Random exploration (15% base, 40% during crisis)
10. **Reflection** -- Every 5 steps, agents summarize learning and adjust strategy

## Collaboration

When 3+ agents agree on the same action, collaboration is detected.
All agents receive a small bonus reward (+0.01), incentivizing consensus.
