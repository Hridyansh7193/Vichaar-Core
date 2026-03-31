from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class Metrics(BaseModel):
    cost: float = 0.0
    expected_profit: float = 0.0
    legal_risk: float = 0.0
    env_impact: float = 0.0
    public_sentiment: float = 0.0

class Uncertainty(BaseModel):
    confidence_scores: Dict[str, float] = Field(default_factory=dict)

class AgentAction(BaseModel):
    agent_name: str
    action_content: str
    metrics_update: Optional[Metrics] = None
    uncertainty_update: Optional[Uncertainty] = None

class State(BaseModel):
    scenario: str = ""
    metrics: Metrics = Field(default_factory=Metrics)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    history: List[AgentAction] = Field(default_factory=list)
