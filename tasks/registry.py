"""Task registry — all scenario definitions."""
from typing import Dict, Any

from tasks.easy import TASK as EASY
from tasks.medium import TASK as MEDIUM
from tasks.hard import TASK as HARD
from tasks.adversarial import TASK as ADVERSARIAL
from tasks.chaotic import TASK as CHAOTIC

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD,
}
