# Vichaar Agentic Decision System

A multi-agent evaluation environment for simulating business decisions with conflicting ethical, legal, and financial priorities.

## 🚀 Quick Start

### 1. Configure API
Create a `.env` file with your OpenRouter or OpenAI keys:
```env
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o-mini
```

### 2. Run Evaluation
Run the full suite of tasks (easy, medium, hard):
```bash
python inference.py
```

### 3. API Demo
Start the FastAPI server:
```bash
python app.py
```
Then POST to `http://127.0.0.1:8000/run`.

## 🏛️ Architecture
- **env.py**: OpenEnv compliant environment wrapper.
- **multi_agent.py**: Orchestrator for Profit, PR, and Ethics LLM agents.
- **grader.py**: Normalized reward engine (0.0 - 1.0).
- **tasks.py**: Predefined scenario suite.
