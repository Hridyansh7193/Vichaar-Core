"""CEO -- Hard Constraint Enforcer. Overrides all decisions when thresholds breach."""
from typing import Dict, Tuple
from configs.env_config import CEO_THRESHOLDS


class CEO:
    """Overrides all agent decisions when critical thresholds are breached.

    Priority: cost > legal > env (handles worst crisis first).
    """

    @staticmethod
    def check(metrics: Dict[str, float]) -> Tuple[str | None, str]:
        """Return (forced_action, reason) or (None, '')."""
        for metric, cfg in CEO_THRESHOLDS.items():
            val = metrics.get(metric, 0.0)
            if val >= cfg["threshold"]:
                return (
                    cfg["force_action"],
                    f"CEO OVERRIDE [{cfg['label']}]: {metric}={val:.2f} >= {cfg['threshold']} -> forcing {cfg['force_action']}"
                )
        return None, ""
