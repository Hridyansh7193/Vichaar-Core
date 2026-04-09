"""Task registry — all scenario definitions."""
from typing import Dict, Any

from tasks.easy import TASK as EASY
from tasks.medium import TASK as MEDIUM
from tasks.hard import TASK as HARD
from tasks.adversarial import TASK as ADVERSARIAL
from tasks.chaotic import TASK as CHAOTIC

TASKS: Dict[str, Dict[str, Any]] = {
    "task_easy": EASY,
    "task_medium": MEDIUM,
    "task_hard": HARD,
    "task_adversarial": ADVERSARIAL,
    "task_chaotic": CHAOTIC,
}
