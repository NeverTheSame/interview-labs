"""Typed contracts for workflow runs. These cross every boundary: the chat
router builds a :class:`RunRequest`, the run service starts it, Temporal
serialises it, the panel renders :class:`WorkflowRun`, and the evals assert on
all of them. Change them deliberately."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

WorkflowKind = Literal["stats"]
WORKFLOW_KINDS: tuple[WorkflowKind, ...] = ("stats",)

# Temporal workflow type names, as they appear in Temporal Cloud. Keep stable.
WORKFLOW_TYPES: dict[str, str] = {
    "stats": "R3StatsWorkflow",
}
KIND_BY_TYPE: dict[str, str] = {v: k for k, v in WORKFLOW_TYPES.items()}


# --- parameters ---------------------------------------------------------------


class StatsParams(BaseModel):
    """A statistics snapshot over the last ``days`` days."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["stats"] = "stats"
    days: int = Field(default=30, ge=1, le=365)


RunParams = StatsParams


class RunRequest(BaseModel):
    """Everything needed to start one run. ``requested_by``/``source``/``model``
    go into the Temporal memo so Temporal Cloud shows who and what started it."""

    model_config = ConfigDict(extra="forbid")
    params: RunParams
    requested_by: str = Field(default="controller", max_length=80)
    # The account the run belongs to; None for runs nobody signed in started
    # (schedules, sign-in off). Set by the server, never taken from the client.
    user_id: int | None = None
    source: str = Field(default="chat", max_length=40)  # chat | panel | schedule | api
    model: str = Field(default="", max_length=160)  # the chat model that routed it, if any
    note: str = Field(default="", max_length=300)  # the user's words, for the run card

    @property
    def kind(self) -> WorkflowKind:
        return self.params.kind


# --- progress, runs, statistics -------------------------------------------------

RunStatus = Literal["queued", "running", "completed", "failed", "cancelled", "terminated", "timed_out", "unknown"]
OPEN_STATUSES: frozenset[str] = frozenset({"queued", "running"})


class RunProgress(BaseModel):
    """What a running workflow says it is doing, via a workflow query."""

    stage: str = ""
    message: str = ""
    done: int = 0
    total: int = 0
    updated_at: datetime | None = None


class WorkflowRun(BaseModel):
    """One run as the panel and the API see it, whichever runner produced it."""

    id: str
    kind: WorkflowKind
    status: RunStatus
    params: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime
    closed_at: datetime | None = None
    progress: RunProgress | None = None
    result: dict[str, Any] | None = None
    error: str = ""
    url: str = ""  # Temporal UI link when known
    backend: Literal["temporal", "local"] = "temporal"
    requested_by: str = ""
    user_id: int | None = None  # the owning account; None for ownerless runs (schedules, sign-in off, old runs)
    source: str = ""
    model: str = ""
    note: str = ""

    @property
    def open(self) -> bool:
        return self.status in OPEN_STATUSES

    @property
    def duration_s(self) -> float | None:
        if self.closed_at is None:
            return None
        return (self.closed_at - self.started_at).total_seconds()


class RunStats(BaseModel):
    """Run statistics over a window, from Temporal visibility (or the local log)."""

    window_days: int
    total: int = 0
    running: int = 0
    failed: int = 0
    by_kind: dict[str, dict[str, int]] = Field(default_factory=dict)  # kind -> status -> count
    avg_duration_s: dict[str, float] = Field(default_factory=dict)  # kind -> seconds
    last_run_at: dict[str, datetime | None] = Field(default_factory=dict)
    recent: list[WorkflowRun] = Field(default_factory=list)


class RunsInfo(BaseModel):
    """How runs execute right now, for the panel header and the setup hints."""

    backend: Literal["temporal", "local", "none"]
    configured: bool
    namespace: str = ""
    task_queue: str = ""
    ui_url: str = ""
    reason: str = ""  # why runs are unavailable or local


# --- results ------------------------------------------------------------------


class CountItem(BaseModel):
    value: str
    count: int


class StatsSnapshot(BaseModel):
    """Statistics as the stats workflow returns them and the panel shows them."""

    kind: Literal["stats"] = "stats"
    captured_at: datetime
    window_days: int = 30
    total_jobs: int = 0
    new_jobs: int = 0
    applied_jobs: int = 0
    jobs_last7: int = 0
    jobs_last30: int = 0
    last_scrape_at: datetime | None = None  # when the newest job arrived
    statuses: dict[str, int] = Field(default_factory=dict)
    top_roles: list[CountItem] = Field(default_factory=list)
    top_locations: list[CountItem] = Field(default_factory=list)
    runs: RunStats | None = None


RunResult = StatsSnapshot


# --- the right-hand panel ----------------------------------------------------------


class Suggestion(BaseModel):
    """One card that nudges the operator: run something now, put a prompt in
    the chat, or open a portal page. Built from deterministic rules over the
    database and recent runs, never from a model."""

    id: str
    title: str
    detail: str = ""
    kind: Literal["run", "prompt", "link"]
    request: RunRequest | None = None  # kind == run: posted to /api/runs
    prompt: str = ""  # kind == prompt: dropped into the chat composer
    href: str = ""  # kind == link: a portal path, relative to the base path
    tone: Literal["accent", "warn", "neutral"] = "neutral"


class PanelState(BaseModel):
    """Everything the right-hand panel needs in one fetch."""

    runs_info: RunsInfo
    facts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # pipeline key -> [{value,label,warn}]
    suggestions: list[Suggestion] = Field(default_factory=list)
    recent_runs: list[WorkflowRun] = Field(default_factory=list)
    stats: StatsSnapshot | None = None
