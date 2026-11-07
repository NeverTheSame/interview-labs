"""The steps of each pipeline, written once. A flow decides which
activities run, in what order, how progress reads and how a failure ends
the run; a :class:`Steps` implementation executes the activities. Temporal
workflows (``workflows``) and the in-process runner (``local``) both drive
these functions, so a run behaves the same under either.

Flows are deterministic: no clock, no randomness, no I/O. Everything with a
side effect is an activity (``activities``); the runner stamps progress
with its own clock (``workflow.now()`` under Temporal)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Awaitable, Callable, Protocol, TypeVar

from pydantic import BaseModel

from .models import RunRequest, StatsParams, StatsSnapshot

T = TypeVar("T")

# ApplicationError types. A fatal error is not retried and stops the run; a
# run failure carries the message the operator sees.
FATAL_ERROR_TYPE = "PipelineUnavailable"
RUN_FAILED_TYPE = "R3RunFailed"


@dataclass(frozen=True)
class Step:
    """One activity: its registered name, what it is called in messages,
    and how long it may take."""

    name: str
    label: str
    timeout: timedelta
    heartbeat: timedelta | None = None
    attempts: int = 3


COMPUTE_STATS = Step("r3_compute_stats", "the statistics", timedelta(minutes=1))
STEPS: tuple[Step, ...] = (COMPUTE_STATS,)

# The workflow query that answers a run's progress.
PROGRESS_QUERY = "progress"

# How long the whole run may take, per kind.
EXECUTION_TIMEOUTS: dict[str, timedelta] = {
    "stats": timedelta(minutes=30),
}


class StepFailed(Exception):
    """An activity failed for good (after its retries). ``fatal`` marks a
    step that cannot work at all, which stops the run."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.fatal = fatal


class FlowFailed(Exception):
    """The run cannot finish. The message is what the operator reads."""


class Steps(Protocol):
    async def run_activity(self, step: Step, arg: Any, result_type: type[T]) -> T: ...


SetProgress = Callable[[str, str, int, int], None]  # stage, message, done, total
Flow = Callable[[RunRequest, Steps, SetProgress], Awaitable[BaseModel]]


# --- activity inputs and outputs ------------------------------------------------------


class StatsJob(BaseModel):
    days: int = 30
    # The run's account: its own job counts. None (sign-in off, schedules)
    # is the whole-portal view.
    user_id: int | None = None


# --- helpers -------------------------------------------------------------------------


def human_duration(td: timedelta | None) -> str:
    """``40 min``, ``90 s``, ``1 h``."""
    seconds = int(td.total_seconds()) if td is not None else 0
    if seconds >= 3600 and seconds % 3600 == 0:
        return f"{seconds // 3600} h"
    if seconds >= 60 and seconds % 60 == 0:
        return f"{seconds // 60} min"
    return f"{seconds} s"


def run_summary(request: RunRequest) -> str:
    """One line for the Temporal UI and the logs."""
    what = f"Statistics over {request.params.days} days"
    return f"{what} · {request.source} · {request.requested_by}"


P = TypeVar("P", bound=BaseModel)


def _params(request: RunRequest, expected: type[P]) -> P:
    """The request's parameters, which must match the workflow. (A mismatch
    can only come from a hand-started workflow; it fails the run instead of
    raising inside workflow code, which Temporal would retry forever.)"""
    if not isinstance(request.params, expected):
        raise FlowFailed(f"This workflow takes {expected.__name__}, not {type(request.params).__name__}")
    return request.params


# --- flows ---------------------------------------------------------------------------


async def stats_flow(request: RunRequest, steps: Steps, progress: SetProgress) -> StatsSnapshot:
    """Snapshot the database's numbers."""
    params = _params(request, StatsParams)
    progress("computing", f"Computing statistics over {params.days} days", 0, 1)
    try:
        snapshot = await steps.run_activity(COMPUTE_STATS, StatsJob(days=params.days, user_id=request.user_id), StatsSnapshot)
    except StepFailed as e:
        raise FlowFailed(f"Statistics failed: {e.message}") from None
    progress("finished", f"{snapshot.total_jobs} jobs, {snapshot.new_jobs} unreviewed, "
             f"{snapshot.applied_jobs} applied", 1, 1)
    return snapshot


FLOWS: dict[str, Flow] = {"stats": stats_flow}
