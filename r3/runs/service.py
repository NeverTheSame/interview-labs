"""``RunService`` is the one interface the web layer and the chat use to
start and observe runs. Two implementations exist: Temporal (production, the
runs live in Temporal Cloud; ``r3.runs.temporal``) and an in-process runner
for local development and tests (``r3.runs.local``). ``build_run_service``
picks one from the settings."""

from __future__ import annotations

import logging
import secrets
from datetime import datetime, timezone
from typing import Protocol

from r3.db import Database
from r3.settings import ControllerSettings

from .models import RunRequest, RunsInfo, RunStats, WorkflowKind, WorkflowRun

log = logging.getLogger("r3.runs")


class RunService(Protocol):
    async def start(self, request: RunRequest) -> WorkflowRun: ...

    async def get(self, run_id: str) -> WorkflowRun | None: ...

    async def list(self, limit: int = 20, kind: WorkflowKind | None = None, *,
                   user_id: int | None = None) -> list[WorkflowRun]: ...
    """``user_id`` keeps only that account's runs; None means every run."""

    async def stats(self, days: int = 30, *, user_id: int | None = None) -> RunStats: ...

    async def cancel(self, run_id: str) -> bool: ...

    def describe(self) -> RunsInfo: ...

    async def close(self) -> None: ...


class RunsUnavailable(Exception):
    """Raised by ``start`` when no runner can take the request."""


class NoRuns:
    """The runner when runs are switched off or nothing is configured."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def start(self, request: RunRequest) -> WorkflowRun:
        raise RunsUnavailable(self.reason)

    async def get(self, run_id: str) -> WorkflowRun | None:
        return None

    async def list(self, limit: int = 20, kind: WorkflowKind | None = None, *,
                   user_id: int | None = None) -> list[WorkflowRun]:
        return []

    async def stats(self, days: int = 30, *, user_id: int | None = None) -> RunStats:
        return RunStats(window_days=days)

    async def cancel(self, run_id: str) -> bool:
        return False

    def describe(self) -> RunsInfo:
        return RunsInfo(backend="none", configured=False, reason=self.reason)

    async def close(self) -> None:
        return None


def new_run_id(kind: str, now: datetime | None = None) -> str:
    """``r3-<kind>-<UTC timestamp>-<6 hex>``: sortable, readable in the
    Temporal UI, unique without coordination."""
    now = now or datetime.now(timezone.utc)
    return f"r3-{kind}-{now.astimezone(timezone.utc):%Y%m%dT%H%M%SZ}-{secrets.token_hex(3)}"


async def build_run_service(settings: ControllerSettings, database: Database) -> RunService:
    """Connect the runner the settings ask for (``R3_RUNS_BACKEND``):

    - ``none``: runs are off.
    - ``temporal``: Temporal or nothing; an unreachable server leaves runs
      unavailable with the reason.
    - ``auto`` (default): Temporal when ``TEMPORAL_ADDRESS`` is set, falling
      back to the in-process runner when it cannot be reached; in-process
      when it is not set.
    - ``local``: always in-process.
    """
    # Imported here: the runners import the activities, which import the
    # statistics module, which imports this one.
    from .local import LocalRuns

    mode = settings.runs_backend
    if mode == "none":
        return NoRuns("Runs are switched off (R3_RUNS_BACKEND=none).")
    if mode == "temporal" or (mode == "auto" and settings.temporal_configured):
        if not settings.temporal_configured:
            return NoRuns("R3_RUNS_BACKEND=temporal but TEMPORAL_ADDRESS is not set.")
        from .temporal import TemporalRuns

        try:
            runs = await TemporalRuns.connect(settings)
        except Exception as e:
            reason = f"Temporal unreachable at {settings.temporal_address}: {type(e).__name__}: {e}".rstrip(": ")
            log.warning("%s", reason)
            if mode == "temporal":
                return NoRuns(reason)
            return LocalRuns(database, settings, reason=(
                f"{reason}. Runs execute inside the portal process until Temporal is reachable and the portal restarts."))
        log.info("runs: Temporal %s namespace=%s task_queue=%s", settings.temporal_address,
                 settings.temporal_namespace, settings.temporal_task_queue)
        return runs
    log.info("runs: in-process")
    return LocalRuns(database, settings)
