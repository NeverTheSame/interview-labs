"""Activities: every side effect a run has. Each is an ``@activity.defn``
method on :class:`Activities`, which holds the database. The Temporal worker
registers ``Activities(...).all()``; the in-process runner calls the same
methods directly through the same flows."""

from __future__ import annotations

import contextvars
from typing import Any, Callable

from temporalio import activity

from r3.db import Database

from .flows import COMPUTE_STATS, StatsJob
from .models import RunProgress, StatsSnapshot
from .stats import compute_stats_snapshot, for_account

ProgressSink = Callable[[RunProgress], None]

# The in-process runner puts a sink here for the task that runs a flow, so
# an activity called outside Temporal still reports its progress.
LOCAL_PROGRESS: contextvars.ContextVar[ProgressSink | None] = contextvars.ContextVar("r3_local_progress", default=None)


class Activities:
    """The activities of every workflow, bound to their dependencies."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def all(self) -> list[Callable[..., Any]]:
        """Everything a worker registers."""
        return list(self.by_name().values())

    def by_name(self) -> dict[str, Callable[..., Any]]:
        return {
            COMPUTE_STATS.name: self.compute_stats,
        }

    # --- stats --------------------------------------------------------------------------

    @activity.defn(name=COMPUTE_STATS.name)
    async def compute_stats(self, request: StatsJob) -> StatsSnapshot:
        """The snapshot; a signed-in account's run carries its own counts."""
        snapshot = await compute_stats_snapshot(self.database, None, request.days, user_id=request.user_id)
        return snapshot if request.user_id is None else for_account(snapshot)
