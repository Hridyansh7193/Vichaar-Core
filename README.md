# Vichaar-Core: OpenEnv Multi-Agent Simulation

Vichaar-Core is a multi-agent reinforcement learning (RL) simulation environment compliant with the OpenEnv specification. It simulates complex business scenarios where profitability, ethics, legal risk, and public sentiment frequently conflict, requiring a balanced decision.

## Environment Description
The environment represents a company facing scenarios like a "Software Update" (easy), a "Personalized Targeting Ad Campaign" (medium), or "Resource Extraction in an ecological region" (hard). Throughout the simulation, 5 specialized LLM-backed agents (Profit, PR, Ethics, Legal, Risk) analyze the situation and vote on the best course of action.

## Action Space
The environment expects a single discrete action string at each step. Valid actions are:
- `increase_production`
- `delay_launch`
- `invest_in_safety`
- `launch_fast`
- `reduce_cost`

## Observation Space
The observation returned by the environment is a strictly typed JSON object containing:
- `scenario` (string): Narrative context of the task.
- `metrics` (object): Normalized metrics between 0-1 for `expected_profit`, `legal_risk`, `env_impact`, `public_sentiment`, and `cost`.
- `history` (list): The list of past selected actions.
- `step_count` (int): The current integer step.

## Reward Logic
After every step, a numeric reward between `0.0` and `1.0` is computed based on the formula:
`reward = 0.4 * expected_profit - 0.3 * legal_risk - 0.2 * env_impact + 0.1 * public_sentiment`

The reward is dynamically adjusted based on task constraints. For example, in hard tasks, high legal risk strongly penalizes the score.

## Setup Instructions

### Local Environment
1. Ensure Python 3.11+ is installed.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Set your API Key in `.env`:
   `OPENAI_API_KEY="your-key-here"`

### Running the API (Uvicorn)
Start the FastAPI server:
`uvicorn app:app --host 0.0.0.0 --port 8000 --reload`

API Endpoints:
- `POST /reset`: Initialize the environment for a specific task.
- `POST /step`: Trigger agents to generate an action and advance the simulation.
- `POST /run`: Automatically run a task simulation for maximum steps.

### Running with Docker
Build the docker image:
`docker build -t vichaar-core .`
Run the container:
`docker run -p 8000:8000 -e OPENAI_API_KEY="your-key-here" vichaar-core`

### Inference Test
Run the inference file to quickly simulate tasks locally:
`python inference.py`
