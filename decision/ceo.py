"""CEO -- Hard Constraint Enforcer. Overrides all decisions when thresholds breach."""
from typing import Dict, Tuple
from configs.env_config import CEO_THRESHOLDS


class CEO:
    """CEO acts as a constraint signal, applying soft penalties rather than hard dictionary overrides."""

    @staticmethod
    def check(metrics: Dict[str, float]) -> Tuple[None, str]:
        """Disabled CEO overrides per request."""
        return None, ""
