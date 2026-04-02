from typing import Dict, Any, Tuple
from tasks import TASKS
import copy
class Env:
    def __init__(self):
        # STRICT DEBUG: Use ONLY self._state, ensure it always contains default values.
        self._state: Dict[str, Any] = {
            "scenario": "Uninitialized",
            "metrics": {
                "expected_profit": 0.5,
                "legal_risk": 0.0,
                "env_impact": 0.0,
                "public_sentiment": 0.5,
                "cost": 0.0
            },
            "history": [],
            "step_count": 0
        }
        self.task_id: str = "medium"

    def state(self) -> Dict[str, Any]:
        return self._state

    def reset(self, task_id: str = "medium") -> Dict[str, Any]:
        """
        Loads scenario, metrics, initializes history/count into self.state.
        """
        if task_id not in TASKS:
             task_id = "medium"
        
        self.task_id = task_id
        task = TASKS[task_id]
        
        self._state = {
            "scenario": task.get("scenario", "Default scenario"),
            "metrics": task.get("metrics", {}).copy(),
            "history": [],
            "step_count": 0
        }
        # Ensure base metrics exist after reset to prevent KeyError
        metrics = self._state["metrics"]
        for key in ["expected_profit", "legal_risk", "env_impact", "public_sentiment", "cost"]:
            if key not in metrics:
                metrics[key] = 0.5 if key in ["expected_profit", "public_sentiment"] else 0.0
                
        return self._state

    def _clamp(self, val: float) -> float:
        """Clamp all values between 0 and 1 and round to 2 decimals."""
        return round(max(0.0, min(1.0, float(val))), 2)

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Applies action deterministic effects on metrics. Updates history.
        """
        import copy
        
        # Save previous metrics
        prev_metrics = copy.deepcopy(self._state.get("metrics", {}))
        
        metrics = self._state.get("metrics", {})
        
        # Safe access to avoid KeyErrors
        profit = metrics.get("expected_profit", 0.0)
        legal = metrics.get("legal_risk", 0.0)
        env_imp = metrics.get("env_impact", 0.0)
        sentiment = metrics.get("public_sentiment", 0.0)
        cost = metrics.get("cost", 0.0)
        
        if action == "increase_production":
            profit += 0.2
            legal += 0.2
            env_imp += 0.1
            sentiment -= 0.1
            cost += 0.1
        elif action == "delay_launch":
            sentiment += 0.2
            profit -= 0.1
            legal -= 0.1
            env_imp -= 0.05
            cost += 0.1
        elif action == "invest_in_safety":
            profit -= 0.1
            legal -= 0.3
            env_imp -= 0.2
            sentiment += 0.1
            cost += 0.2
        elif action == "launch_fast":
            profit += 0.3
            legal += 0.3
            sentiment -= 0.2
            env_imp += 0.1
            cost -= 0.1
        elif action == "reduce_cost":
            cost -= 0.2
            profit += 0.1
            env_imp += 0.2
            legal += 0.1
            sentiment -= 0.1
            
        # Reassign and Clamp
        metrics["expected_profit"] = self._clamp(profit)
        metrics["legal_risk"] = self._clamp(legal)
        metrics["env_impact"] = self._clamp(env_imp)
        metrics["public_sentiment"] = self._clamp(sentiment)
        metrics["cost"] = self._clamp(cost)
            
        self._state["metrics"] = metrics
        
        # Update history safely
        history = self._state.get("history", [])
        history.append(action)
        self._state["history"] = history[-10:]
        
        self._state["step_count"] = self._state.get("step_count", 0) + 1
            
        # Calculate Reward based on metric change
        current_reward = self.compute_step_reward(prev_metrics, metrics)
        
        # Determine simulation end status
        done = self._state["step_count"] >= 3  
        
        info = {}
        
        return self._state, current_reward, done, info
        
    def compute_step_reward(self, prev_metrics: Dict[str, Any], current_metrics: Dict[str, Any]) -> float:
        reward = 0.0

        # reward improvement
        reward += 0.2 * (float(current_metrics.get("expected_profit", 0.0)) - float(prev_metrics.get("expected_profit", 0.0)))

        # penalize increase in risk
        reward -= 0.2 * (float(current_metrics.get("legal_risk", 0.0)) - float(prev_metrics.get("legal_risk", 0.0)))
        reward -= 0.2 * (float(current_metrics.get("env_impact", 0.0)) - float(prev_metrics.get("env_impact", 0.0)))

        # reward sentiment improvement
        reward += 0.1 * (float(current_metrics.get("public_sentiment", 0.0)) - float(prev_metrics.get("public_sentiment", 0.0)))

        return round(max(-1.0, min(1.0, float(reward))), 2)
