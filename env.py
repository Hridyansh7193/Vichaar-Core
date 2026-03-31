import asyncio
from typing import Dict, Any, Tuple, Optional
from tasks import TASKS
from multi_agent import profit_agent, pr_agent, ethics_agent, final_agent
from grader import compute_reward, get_intermediate_reward
from openenv.models import Observation, EnvAction, RewardResponse
import config

class Env:
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._task_id: str = ""
        self._round_num: int = 0
        self._max_rounds: int = 2
        self._done: bool = False

    def reset(self, task_id: str = "easy") -> Observation:
        """Resets the environment with a specific task_id."""
        task = TASKS.get(task_id, TASKS["easy"])
        self._task_id = task_id
        self._state = {
            "id": task["id"],
            "description": task["description"],
            "scenario": task["scenario"],
            "metrics": task["metrics"].copy(),
            "uncertainty": task["uncertainty"].copy(),
            "history": [] 
        }
        self._round_num = 0
        self._done = False
        return self._get_observation()

    async def _run_agents(self) -> float:
        """Run a full round of DELIBERATION sub-agents (Profit, PR, Ethics)."""
        agents = [profit_agent, pr_agent, ethics_agent]
        round_actions = []
        
        # Sequentially collect opinions
        for agent_fn in agents:
            action_output = await agent_fn(self._state)
            round_actions.append(action_output)
            
        # Append to state history and limit size
        self._state["history"].extend(round_actions)
        if len(self._state["history"]) > 10:
             self._state["history"] = self._state["history"][-10:]
             
        self._round_num += 1
        
        # Calculate intermediate reward for this round
        return get_intermediate_reward(self._state, self._round_num)

    async def step(self, action: Dict[str, Any]) -> RewardResponse:
        """
        Executes one step in the OpenEnv flow.
        action = {"action_type": "step" | "finalize"}
        """
        if self._done:
             return RewardResponse(reward=0.0, done=True, info={"error": "Episode finished."})

        # Validate action
        env_action = EnvAction(**action)
        action_type = env_action.action_type
        
        reward = 0.0
        info = {}

        if action_type == "step":
            if self._round_num < self._max_rounds:
                reward = await self._run_agents()
                info = {"status": f"Round {self._round_num} complete."}
            else:
                action_type = "finalize"

        if action_type == "finalize":
            final_decision_output = await final_agent(self._state)
            self._done = True
            reward = compute_reward(self._state, final_decision_output, self._task_id)
            info = {"status": "Decision finalized.", "final_decision": final_decision_output}

        return RewardResponse(reward=float(reward), done=self._done, info=info)

    def state(self) -> Dict[str, Any]:
        """Returns the raw environment state."""
        return self._state.copy()

    def _get_observation(self) -> Observation:
        """Returns the public observation (scenario, metrics, history)."""
        return Observation(
            scenario=self._state.get("scenario", ""),
            metrics=self._state.get("metrics", {}),
            history=self._state.get("history", [])
        )

    @property
    def round_num(self):
        return self._round_num

    @property
    def done(self):
        return self._done
