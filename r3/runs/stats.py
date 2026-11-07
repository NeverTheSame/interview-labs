"""The statistics snapshot: the database numbers every stats surface shares
(the stats workflow, ``GET /api/stats``, the chat's stats action, the panel),
plus run statistics when a run service is at hand."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from r3.db import Database, Facets, Stats

from .models import CountItem, RunStats, StatsSnapshot, WorkflowRun
from .service import RunService

RUN_STATS_TIMEOUT = 8.0


async def run_stats_or_none(runs: RunService | None, days: int, *, user_id: int | None = None) -> RunStats | None:
    """Run statistics (one account's when ``user_id`` is given), or None when
    there is no run service or it does not answer in time. A failing run
    service never fails a snapshot."""
    if runs is None:
        return None
    try:
        async with asyncio.timeout(RUN_STATS_TIMEOUT):
            return await (runs.stats(days) if user_id is None else runs.stats(days, user_id=user_id))
    except Exception:
        return None


def hide_run_internals(run: WorkflowRun) -> WorkflowRun:
    """A run as a regular account sees it: no Temporal UI link."""
    return run.model_copy(update={"url": ""}) if run.url else run


def for_account(snapshot: StatsSnapshot) -> StatsSnapshot:
    """The snapshot as a regular account sees it: no Temporal links."""
    runs = snapshot.runs
    if runs is None:
        return snapshot
    return snapshot.model_copy(update={"runs": runs.model_copy(update={"recent": [hide_run_internals(r) for r in runs.recent]})})


def job_queries(database: Database, user_id: int | None = None) -> tuple[Any, Any]:
    """The overview and facet queries, one account's job counts when
    ``user_id`` is given."""
    if user_id is None:
        return database.overview(), database.job_facets()
    return database.overview(user_id=user_id), database.job_facets(user_id=user_id)


def build_stats_snapshot(stats: Stats, facets: Facets, days: int = 30, run_stats: RunStats | None = None,
                         captured_at: datetime | None = None) -> StatsSnapshot:
    """Assemble the snapshot from the overview and facet queries (the same
    ones the overview page uses)."""
    return StatsSnapshot(
        captured_at=captured_at or datetime.now(timezone.utc),
        window_days=days,
        total_jobs=stats.total_jobs,
        new_jobs=stats.new_jobs,
        applied_jobs=stats.applied_jobs,
        jobs_last7=stats.jobs_last7,
        jobs_last30=stats.jobs_last30,
        last_scrape_at=stats.last_scrape_at,
        statuses={v.value: v.count for v in facets.statuses},
        top_roles=[CountItem(value=v.value, count=v.count) for v in facets.role_categories[:8]],
        top_locations=[CountItem(value=v.value, count=v.count) for v in facets.location_kinds[:8]],
        runs=run_stats,
    )


async def compute_stats_snapshot(database: Database, runs: RunService | None = None, days: int = 30, *,
                                 user_id: int | None = None) -> StatsSnapshot:
    """Build the snapshot from the same queries the overview page uses, so
    every number matches what it shows. Run statistics are attached when
    ``runs`` is given and answers; a failing run service never fails the
    snapshot. With a ``user_id`` the job counts are that account's own
    triage and the run statistics its own runs."""
    (stats, facets), run_stats = await asyncio.gather(
        asyncio.gather(*job_queries(database, user_id)),
        run_stats_or_none(runs, days, user_id=user_id),
    )
    return build_stats_snapshot(stats, facets, days, run_stats)
