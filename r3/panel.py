"""The Controller's right-hand panel in one fetch (``GET /api/panel``): how
runs execute, the pipelines' live facts, suggestions, recent runs and the
statistics snapshot. Suggestions come from fixed rules over the snapshot
and the recent runs, never from a model, so they are cheap, repeatable and
testable. Every part is fetched on its own guard: a database or Temporal
outage blanks only the parts that depend on it."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from r3.db import Database
from r3.pipelines import pipeline_facts
from r3.runs.models import (
    PanelState,
    RunRequest,
    RunsInfo,
    StatsParams,
    StatsSnapshot,
    Suggestion,
    WorkflowRun,
)
from r3.runs.service import RunService
from r3.runs.stats import build_stats_snapshot, for_account, hide_run_internals, job_queries, run_stats_or_none
from r3.settings import ControllerSettings

log = logging.getLogger("r3.panel")

MAX_SUGGESTIONS = 6
RECENT_RUNS = 8
PANEL_TIMEOUT = 8.0
DAY = timedelta(hours=24)
TONE_ORDER = {"warn": 0, "accent": 1, "neutral": 2}
FAILED = frozenset({"failed", "timed_out"})
# Suggestions that point at operator pages (runner setup); a regular account
# never sees them.
OPERATOR_SUGGESTIONS = frozenset({"connect-temporal"})
# A regular account's panel: the job-seeker names for its facts, without the
# corpus ones.
SEEKER_FACT_LABELS = {"unreviewed": "new for you", "applied": "applied by you", "skipped": "passed on"}
OPERATOR_FACTS = frozenset({"jobs stored", "added 7d"})


def _last_stats_run(snapshot: StatsSnapshot, recent: list[WorkflowRun]) -> datetime | None:
    seen = [r.started_at for r in recent if r.kind == "stats" and r.status not in FAILED]
    if snapshot.runs is not None and (at := snapshot.runs.last_run_at.get("stats")) is not None:
        seen.append(at)
    return max(seen) if seen else None


def build_suggestions(snapshot: StatsSnapshot, runs_info: RunsInfo, recent: list[WorkflowRun]) -> list[Suggestion]:
    """The panel's nudges, from the snapshot and the recent runs. Pure: "now"
    is the snapshot's capture time. Ordered warn, then accent, then neutral,
    at most six. Run suggestions are offered only when runs can start."""
    now = snapshot.captured_at
    can_run = runs_info.configured and runs_info.backend != "none"
    out: list[Suggestion] = []

    failed = sorted((r for r in recent if r.status in FAILED and (r.closed_at or r.started_at) >= now - DAY),
                    key=lambda r: r.started_at, reverse=True)
    if failed:
        run = failed[0]
        out.append(Suggestion(id=f"failed-{run.id}", title=f"The last {run.kind} run failed",
                              detail=(run.error or "Open the run for details")[:240], kind="link",
                              href=f"/controller#run={run.id}", tone="warn"))

    if snapshot.new_jobs > 0:
        out.append(Suggestion(id="review-new", title=f"Review {snapshot.new_jobs:,} unreviewed "
                              f"{'job' if snapshot.new_jobs == 1 else 'jobs'}",
                              detail="Waiting for triage", kind="link", href="/jobs?status=new",
                              tone="accent"))
        out.append(Suggestion(id="top3", title="Show the top 3 newest jobs",
                              detail="The assistant lists them with a line on each", kind="prompt",
                              prompt="Show me the top 3 newest unreviewed jobs", tone="neutral"))

    last_stats = _last_stats_run(snapshot, recent)
    if can_run and (last_stats is None or last_stats < now - DAY):
        out.append(Suggestion(id="stats-today", title="Snapshot today's statistics",
                              detail="Records today's numbers as a run", kind="run", tone="neutral",
                              request=RunRequest(params=StatsParams(), source="panel",
                                                 note="Snapshot today's statistics")))

    if not runs_info.configured:
        out.append(Suggestion(id="connect-temporal", title="Connect Temporal to run workflows",
                              detail=runs_info.reason, kind="link", href="/controller#setup", tone="neutral"))

    out.sort(key=lambda s: TONE_ORDER[s.tone])
    return out[:MAX_SUGGESTIONS]


async def panel_state(database: Database, runs: RunService, settings: ControllerSettings | None = None,
                      *, days: int = 30, user_id: int | None = None, operator: bool = True) -> PanelState:
    """Assemble the panel. The database queries run once and feed both the
    pipeline facts and the snapshot; the run service's statistics and recent
    runs are fetched beside them. ``settings`` is accepted so callers can
    pass the app's state uniformly; everything else comes from the run
    service. With a ``user_id`` the job counts and runs are that account's
    own; ``operator=False`` also hides operator details (corpus counts,
    setup details, Temporal links) for a regular account."""

    async def db_parts() -> tuple[Any, Any]:
        async with asyncio.timeout(PANEL_TIMEOUT):
            return tuple(await asyncio.gather(*job_queries(database, user_id)))

    async def recent_runs() -> list[WorkflowRun]:
        async with asyncio.timeout(PANEL_TIMEOUT):
            return await (runs.list(RECENT_RUNS) if user_id is None else runs.list(RECENT_RUNS, user_id=user_id))

    parts, run_stats, recent = await asyncio.gather(db_parts(), run_stats_or_none(runs, days, user_id=user_id),
                                                    recent_runs(),
                                                    return_exceptions=True)
    if isinstance(recent, BaseException):
        log.warning("panel: recent runs unavailable: %s", recent)
        recent = []
    if isinstance(run_stats, BaseException):
        run_stats = None

    facts: dict[str, list[dict[str, Any]]] = {}
    snapshot: StatsSnapshot | None = None
    if isinstance(parts, BaseException):
        log.warning("panel: database unavailable: %s", type(parts).__name__)
    else:
        stats, facets = parts
        try:
            facts = {key: [dataclasses.asdict(f) for f in items]
                     for key, items in pipeline_facts(stats, facets).items()}
        except Exception:
            log.exception("panel: pipeline facts failed")
        try:
            snapshot = build_stats_snapshot(stats, facets, days, run_stats)
        except Exception:
            log.exception("panel: statistics snapshot failed")

    try:
        runs_info = runs.describe()
    except Exception:
        log.exception("panel: run service description failed")
        runs_info = RunsInfo(backend="none", configured=False, reason="The run service did not answer")

    if not operator:
        runs_info = runs_info.model_copy(update={"namespace": "", "task_queue": "", "ui_url": "", "reason": ""})
        facts = {key: [{**f, "label": SEEKER_FACT_LABELS.get(f.get("label"), f.get("label"))}
                       for f in items if f.get("label") not in OPERATOR_FACTS]
                 for key, items in facts.items()}
        recent = [hide_run_internals(r) for r in recent]
        if snapshot is not None:
            snapshot = for_account(snapshot)
        elif run_stats is not None:
            run_stats = run_stats.model_copy(update={"recent": [hide_run_internals(r) for r in run_stats.recent]})

    basis = snapshot or StatsSnapshot(captured_at=datetime.now(timezone.utc), window_days=days, runs=run_stats)
    try:
        suggestions = build_suggestions(basis, runs_info, recent)
    except Exception:
        log.exception("panel: suggestions failed")
        suggestions = []
    if not operator:
        kept: list[Suggestion] = []
        for s in suggestions:
            if s.id in OPERATOR_SUGGESTIONS | {"stats-today"}:
                continue
            if s.id == "review-new":
                s = s.model_copy(update={"title": f"Look through your {basis.new_jobs:,} new "
                                         f"{'job' if basis.new_jobs == 1 else 'jobs'}",
                                         "detail": "Jobs you haven't looked at yet"})
            kept.append(s)
        suggestions = kept
    return PanelState(runs_info=runs_info, facts=facts, suggestions=suggestions, recent_runs=recent, stats=snapshot)
