"""Pydantic request/response schemas for the Vichaar-Core API."""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class Metrics(BaseModel):
    expected_profit: float
    legal_risk: float
    env_impact: float
    public_sentiment: float
    cost: float


class Observation(BaseModel):
    scenario: str
    phase: str
    metrics: Metrics
    entities: Dict[str, Any]
    events: List[str]
    history: List[str]
    step_count: int
    agent_messages: List[str]
    metrics_trend: List[Dict[str, float]]


class ResetRequest(BaseModel):
    task_id: str = "medium"


class StepResponse(BaseModel):
    observation: Observation
    action: str
    agent_votes: Dict[str, str]
    reward: float
    done: bool
    info: Dict[str, Any]


class RunRequest(BaseModel):
    task_id: str = "medium"
    max_steps: Optional[int] = None


class RunSummary(BaseModel):
    total_steps: int
    final_grade: float
    total_agent_rewards: Dict[str, float]
    performance: str
    collaborated_steps: int


class RunResponse(BaseModel):
    history: List[Dict[str, Any]]
    final_state: Observation
    summary: RunSummary
