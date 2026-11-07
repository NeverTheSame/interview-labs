"""``LocalRuns``: the run service without Temporal. The same flows run in
asyncio tasks inside the portal process and call the activities directly,
with the same timeouts and retries. Runs live in memory (the newest 200)
and are lost on restart, which the panel says. One run per kind executes at
a time; later ones wait as ``queued``."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, TypeVar

from temporalio.exceptions import ApplicationError

from r3.db import Database
from r3.settings import ControllerSettings

from r3.events import record as record_event

from .activities import LOCAL_PROGRESS, Activities
from .audit import record_run_started
from .flows import FATAL_ERROR_TYPE, FLOWS, FlowFailed, Step, StepFailed, human_duration
from .models import RunProgress, RunRequest, RunsInfo, RunStats, WorkflowKind, WorkflowRun
from .service import new_run_id

log = logging.getLogger("r3.runs.local")

T = TypeVar("T")

LOCAL_REASON = "Runs execute inside the portal process; connect Temporal (TEMPORAL_ADDRESS) to persist them"
MAX_RUNS = 200


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LocalSteps:
    """Runs a flow's steps by calling the activity methods directly, with
    each step's timeout and retry count."""

    def __init__(self, activities: Activities, *, retry_delay: float = 1.0) -> None:
        self._fns = activities.by_name()
        self.retry_delay = retry_delay

    async def run_activity(self, step: Step, arg: Any, result_type: type[T]) -> T:
        fn = self._fns[step.name]
        failure = StepFailed(f"{step.label} did not run")
        for attempt in range(1, step.attempts + 1):
            try:
                async with asyncio.timeout(step.timeout.total_seconds()):
                    return await (fn() if arg is None else fn(arg))
            except TimeoutError:
                failure = StepFailed(f"{step.label} timed out after {human_duration(step.timeout)}")
            except ApplicationError as e:
                if e.non_retryable:
                    raise StepFailed(e.message, fatal=e.type == FATAL_ERROR_TYPE) from None
                failure = StepFailed(e.message)
            except Exception as e:
                log.warning("%s failed (attempt %d of %d)", step.name, attempt, step.attempts, exc_info=True)
                failure = StepFailed(f"{type(e).__name__}: {e}")
            if attempt < step.attempts:
                await asyncio.sleep(self.retry_delay * 2 ** (attempt - 1))
        raise failure


class LocalRuns:
    """The in-process ``RunService``."""

    def __init__(self, database: Database, settings: ControllerSettings | None = None, *,
                 reason: str = "", retry_delay: float = 1.0, max_runs: int = MAX_RUNS) -> None:
        self.database = database
        self.settings = settings
        self.reason = reason or LOCAL_REASON
        self.activities = Activities(database)
        self._steps = LocalSteps(self.activities, retry_delay=retry_delay)
        self._runs: OrderedDict[str, WorkflowRun] = OrderedDict()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._active: dict[str, int] = {}
        self._max_runs = max_runs

    # --- bookkeeping -------------------------------------------------------------------

    def _update(self, run_id: str, **fields: Any) -> None:
        run = self._runs.get(run_id)
        if run is not None:
            self._runs[run_id] = run.model_copy(update=fields)

    def _finish(self, run_id: str, status: str, *, result: dict[str, Any] | None = None, error: str = "",
                stage: str = "") -> None:
        run = self._runs.get(run_id)
        if run is None or not run.open:
            return
        message = str((result or {}).get("message") or error or status)
        record_event("run.finished", f"{run.kind.capitalize()} run {status}: {message}", user_id=run.user_id,
                     subject=f"run:{run_id}", data={"kind": run.kind, "status": status, "message": message,
                                                    "error": error or None, "result": result})
        progress = run.progress or RunProgress()
        if stage:
            progress = progress.model_copy(update={"stage": stage, "message": error or progress.message,
                                                   "updated_at": _now()})
        self._update(run_id, status=status, closed_at=_now(), result=result, error=error, progress=progress)

    def _evict(self) -> None:
        """Keep the newest runs; closed ones go first."""
        while len(self._runs) > self._max_runs:
            victim = next((rid for rid, r in self._runs.items() if not r.open), None)
            if victim is None:
                return
            del self._runs[victim]

    # --- RunService ----------------------------------------------------------------------

    async def start(self, request: RunRequest) -> WorkflowRun:
        kind = request.kind
        now = _now()
        run_id = new_run_id(kind, now)
        self._active[kind] = self._active.get(kind, 0) + 1
        queued = self._active[kind] > 1
        run = WorkflowRun(
            id=run_id, kind=kind, status="queued" if queued else "running",
            params=request.params.model_dump(mode="json"), started_at=now,
            progress=RunProgress(stage="queued" if queued else "starting",
                                 message=f"Waiting for the {kind} run before it to finish" if queued else "Starting",
                                 updated_at=now),
            backend="local", requested_by=request.requested_by, user_id=request.user_id, source=request.source,
            model=request.model, note=request.note,
        )
        self._runs[run_id] = run
        record_run_started(run)
        self._evict()
        lock = self._locks.setdefault(kind, asyncio.Lock())
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, request, lock), name=run_id)
        return run.model_copy(deep=True)

    async def _execute(self, run_id: str, request: RunRequest, lock: asyncio.Lock) -> None:
        def set_progress(stage: str, message: str = "", done: int = 0, total: int = 0) -> None:
            self._update(run_id, progress=RunProgress(stage=stage, message=message, done=done, total=total,
                                                      updated_at=_now()))

        def activity_progress(progress: RunProgress) -> None:
            # An activity refines the flow's current stage with its own
            # message and counts; the stage stays the flow's.
            run = self._runs.get(run_id)
            stage = run.progress.stage if run is not None and run.progress is not None else progress.stage
            self._update(run_id, progress=progress.model_copy(update={"stage": stage}))

        # This task runs in its own context, so the sink is this run's alone.
        LOCAL_PROGRESS.set(activity_progress)
        try:
            async with lock:
                self._update(run_id, status="running")
                result = await FLOWS[request.kind](request, self._steps, set_progress)
        except FlowFailed as e:
            self._finish(run_id, "failed", error=str(e), stage="failed")
        except asyncio.CancelledError:
            self._finish(run_id, "cancelled", stage="cancelled")
            raise
        except Exception as e:
            log.exception("local run %s crashed", run_id)
            self._finish(run_id, "failed", error=f"{type(e).__name__}: {e}", stage="failed")
        else:
            self._finish(run_id, "completed", result=result.model_dump(mode="json"))
        finally:
            self._active[request.kind] = max(0, self._active.get(request.kind, 1) - 1)
            self._tasks.pop(run_id, None)

    async def wait(self, run_id: str, timeout: float = 30.0) -> WorkflowRun | None:
        """Wait for a run to close (tests and scripts); returns it either way."""
        task = self._tasks.get(run_id)
        if task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout)
            except (TimeoutError, asyncio.CancelledError):
                pass
        return await self.get(run_id)

    async def get(self, run_id: str) -> WorkflowRun | None:
        run = self._runs.get(run_id)
        return run.model_copy(deep=True) if run is not None else None

    async def list(self, limit: int = 20, kind: WorkflowKind | None = None, *,
                   user_id: int | None = None) -> list[WorkflowRun]:
        rows = [r for r in self._runs.values()
                if (kind is None or r.kind == kind) and (user_id is None or r.user_id == user_id)]
        rows.sort(key=lambda r: r.started_at, reverse=True)
        return [r.model_copy(deep=True) for r in rows[:max(0, limit)]]

    async def stats(self, days: int = 30, *, user_id: int | None = None) -> RunStats:
        since = _now() - timedelta(days=days)
        by_kind: dict[str, dict[str, int]] = {}
        durations: dict[str, list[float]] = {}
        last: dict[str, datetime | None] = {}
        for run in self._runs.values():
            if user_id is not None and run.user_id != user_id:
                continue
            last[run.kind] = max(filter(None, [last.get(run.kind), run.started_at]))
            if run.started_at < since:
                continue
            counts = by_kind.setdefault(run.kind, {})
            counts[run.status] = counts.get(run.status, 0) + 1
            if run.status == "completed" and run.duration_s is not None:
                durations.setdefault(run.kind, []).append(run.duration_s)
        return RunStats(
            window_days=days,
            total=sum(sum(c.values()) for c in by_kind.values()),
            running=sum(c.get("running", 0) + c.get("queued", 0) for c in by_kind.values()),
            failed=sum(c.get("failed", 0) + c.get("timed_out", 0) for c in by_kind.values()),
            by_kind=by_kind,
            avg_duration_s={k: round(sum(v) / len(v), 3) for k, v in durations.items() if v},
            last_run_at=last,
            recent=await self.list(10, user_id=user_id),
        )

    async def cancel(self, run_id: str) -> bool:
        task = self._tasks.get(run_id)
        run = self._runs.get(run_id)
        if task is None or task.done() or run is None or not run.open:
            return False
        self._finish(run_id, "cancelled", stage="cancelled")
        task.cancel()
        return True

    def describe(self) -> RunsInfo:
        return RunsInfo(backend="local", configured=True, reason=self.reason)

    async def close(self) -> None:
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
