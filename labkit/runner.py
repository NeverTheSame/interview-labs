"""Run a lab's checks: every task gets a result, and one failing or crashing
check never hides the others."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .catalog import Lab, Task


class CheckFailed(Exception):
    """Raised by a check with a message the learner can act on."""


@dataclass
class Result:
    task: Task
    passed: bool
    message: str = ""


def run_checks(lab: Lab, checks: dict[str, Callable], aws: Any, only: str | None = None) -> list[Result]:
    results = []
    for task in lab.tasks:
        if only and task.id != only:
            continue
        try:
            checks[task.check](aws)
        except CheckFailed as e:
            results.append(Result(task, False, str(e)))
        except Exception as e:  # a crash is a failed check, reported, never a traceback
            results.append(Result(task, False, f"{type(e).__name__}: {e}"))
        else:
            results.append(Result(task, True))
    return results
