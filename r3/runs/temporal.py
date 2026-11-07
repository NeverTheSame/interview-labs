"""``TemporalRuns``: the run service over Temporal. Every run is a workflow
(``R3StatsWorkflow``) on the configured task queue, carrying a
memo with who started it and why, so Temporal Cloud is the record of runs;
the ``r3-worker`` process executes them.

Reads never raise: when Temporal is slow or down, ``get``/``list``/``stats``
log and return nothing, so the Controller keeps rendering. Only ``start``
raises (``RunsUnavailable``), because the caller must know the run did not
start."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from temporalio.api.enums.v1 import TaskQueueType
from temporalio.api.taskqueue.v1 import TaskQueue
from temporalio.api.workflowservice.v1 import DescribeTaskQueueRequest
from temporalio.client import (
    Client,
    WorkflowExecution,
    WorkflowExecutionDescription,
    WorkflowExecutionStatus,
    WorkflowFailureError,
    WorkflowHandle,
)
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.exceptions import ApplicationError, CancelledError, TerminatedError
from temporalio.exceptions import TimeoutError as TemporalTimeoutError
from temporalio.service import RPCError, RPCStatusCode, TLSConfig

from r3.settings import ControllerSettings

from .audit import record_run_started
from .flows import EXECUTION_TIMEOUTS, PROGRESS_QUERY, run_summary
from .models import (
    KIND_BY_TYPE,
    WORKFLOW_KINDS,
    WORKFLOW_TYPES,
    RunProgress,
    RunRequest,
    RunsInfo,
    RunStats,
    RunStatus,
    StatsSnapshot,
    WorkflowKind,
    WorkflowRun,
)
from .service import RunsUnavailable, new_run_id

log = logging.getLogger("r3.runs.temporal")

STATUS_MAP: dict[WorkflowExecutionStatus, RunStatus] = {
    WorkflowExecutionStatus.RUNNING: "running",
    WorkflowExecutionStatus.COMPLETED: "completed",
    WorkflowExecutionStatus.FAILED: "failed",
    WorkflowExecutionStatus.CANCELED: "cancelled",
    WorkflowExecutionStatus.TERMINATED: "terminated",
    WorkflowExecutionStatus.TIMED_OUT: "timed_out",
    WorkflowExecutionStatus.CONTINUED_AS_NEW: "running",
}
# ExecutionStatus as visibility's GROUP BY reports it.
VISIBILITY_STATUS: dict[str, RunStatus] = {
    "Running": "running", "Completed": "completed", "Failed": "failed", "Canceled": "cancelled",
    "Terminated": "terminated", "TimedOut": "timed_out", "ContinuedAsNew": "running",
}
RESULT_MODELS: dict[str, type] = {"stats": StatsSnapshot}

RPC_TIMEOUT = 5.0  # one visibility or describe call
QUERY_TIMEOUT = 3.0  # a progress query; a missing worker must not hang the API
RESULT_CACHE_SIZE = 200
LIST_OUTCOMES = 10  # finished runs whose outcome one list call may fetch
LIST_CONCURRENCY = 5
STATS_SAMPLE = 500  # runs read per stats call for durations (counts come from count_workflows)
# The owner lives in the memo, which visibility cannot filter on, so one
# account's runs are found by reading this many recent runs and keeping theirs.
USER_SCAN = 500


def _memo_user_id(memo: dict[str, Any]) -> int | None:
    """The owning account from a run's memo; None for ownerless and older runs."""
    value = memo.get("user_id")
    return value if isinstance(value, int) and not isinstance(value, bool) else None
WORKER_SEEN_TTL = 20.0  # how long "a worker is polling" is trusted
WORKER_UNSEEN_TTL = 5.0  # a worker that just started must show up quickly


def _pem(value: str) -> bytes:
    """A PEM given inline (as secrets usually are) or as a file path."""
    value = value.strip()
    if value.startswith("-----BEGIN"):
        return value.replace("\\n", "\n").encode()
    return Path(value).expanduser().read_bytes()


async def connect_client(settings: ControllerSettings, *, timeout: float = 15.0) -> Client:
    """Connect to the configured namespace with the Pydantic data converter.
    An API key means TLS with a bearer key (Temporal Cloud); a client
    certificate and key mean mTLS; otherwise the connection is plain (the
    local dev server)."""
    if not settings.temporal_address:
        raise RunsUnavailable("TEMPORAL_ADDRESS is not set")
    kwargs: dict[str, Any] = {"namespace": settings.temporal_namespace, "data_converter": pydantic_data_converter}
    if settings.temporal_api_key:
        kwargs.update(api_key=settings.temporal_api_key, tls=True,
                      rpc_metadata={"temporal-namespace": settings.temporal_namespace})
    elif settings.temporal_tls_cert or settings.temporal_tls_key:
        if not (settings.temporal_tls_cert and settings.temporal_tls_key):
            raise RunsUnavailable("Set both TEMPORAL_TLS_CERT and TEMPORAL_TLS_KEY for an mTLS namespace")
        kwargs["tls"] = TLSConfig(client_cert=_pem(settings.temporal_tls_cert),
                                  client_private_key=_pem(settings.temporal_tls_key))
    else:
        kwargs["tls"] = False
    async with asyncio.timeout(timeout):
        return await Client.connect(settings.temporal_address, **kwargs)


def _rpc_message(e: BaseException) -> str:
    if isinstance(e, RPCError):
        return e.message or e.status.name
    if isinstance(e, TimeoutError):
        return "timed out"
    return f"{type(e).__name__}: {e}"


def _failure_message(e: WorkflowFailureError) -> str:
    """What the operator reads for a run that did not complete: the failure's
    message only, never a stack trace."""
    cause = e.cause
    if isinstance(cause, ApplicationError):
        return cause.message
    if isinstance(cause, TerminatedError):
        return f"Terminated: {cause.message}" if cause.message else "Terminated"
    if isinstance(cause, TemporalTimeoutError):
        return "Timed out: the run exceeded its execution timeout"
    if isinstance(cause, CancelledError):
        return "Cancelled"
    return getattr(cause, "message", "") or str(cause or e.message or "Failed")


def _normalize_result(kind: str, value: Any) -> dict[str, Any] | None:
    """A completed run's result as JSON, validated against its model when it
    still fits (an older worker may have returned an older shape)."""
    model = RESULT_MODELS.get(kind)
    if model is not None:
        try:
            return model.model_validate(value).model_dump(mode="json")
        except Exception:
            pass
    if isinstance(value, dict):
        return value
    return None if value is None else {"value": value}


class TemporalRuns:
    """The Temporal ``RunService``."""

    def __init__(self, client: Client, settings: ControllerSettings) -> None:
        self.client = client
        self.settings = settings
        self.task_queue = settings.temporal_task_queue
        self._outcomes: OrderedDict[str, tuple[dict[str, Any] | None, str]] = OrderedDict()
        self._worker_seen: bool | None = None
        self._worker_checked = float("-inf")

    @classmethod
    async def connect(cls, settings: ControllerSettings) -> TemporalRuns:
        return cls(await connect_client(settings), settings)

    # --- helpers ---------------------------------------------------------------------

    def _filter(self, kind: str | None = None) -> str:
        """Our workflow types on our task queue (a namespace may be shared)."""
        types = [WORKFLOW_TYPES[kind]] if kind else [WORKFLOW_TYPES[k] for k in WORKFLOW_KINDS]
        quoted = ", ".join(f"'{t}'" for t in types)
        return f"WorkflowType IN ({quoted}) AND TaskQueue = '{self.task_queue}'"

    def _remember(self, run_id: str, outcome: tuple[dict[str, Any] | None, str]) -> None:
        self._outcomes[run_id] = outcome
        self._outcomes.move_to_end(run_id)
        while len(self._outcomes) > RESULT_CACHE_SIZE:
            self._outcomes.popitem(last=False)

    async def _worker_available(self) -> bool | None:
        """Whether any worker polled our task queue recently (None: unknown).
        Rechecked after 20 s (5 s while none was seen); ``describe`` reports it."""
        ttl = WORKER_SEEN_TTL if self._worker_seen else WORKER_UNSEEN_TTL
        if time.monotonic() - self._worker_checked < ttl:
            return self._worker_seen
        self._worker_checked = time.monotonic()
        try:
            request = DescribeTaskQueueRequest(namespace=self.client.namespace, task_queue=TaskQueue(name=self.task_queue),
                                               task_queue_type=TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW)
            async with asyncio.timeout(QUERY_TIMEOUT + 1):
                response = await self.client.workflow_service.describe_task_queue(
                    request, timeout=timedelta(seconds=QUERY_TIMEOUT))
            self._worker_seen = len(response.pollers) > 0
        except Exception as e:
            log.debug("task queue check failed: %s", _rpc_message(e))
            self._worker_seen = None
        return self._worker_seen

    async def _list_executions(self, query: str, limit: int) -> list[WorkflowExecution]:
        async with asyncio.timeout(RPC_TIMEOUT * 2):
            return [ex async for ex in self.client.list_workflows(
                query, limit=limit, page_size=min(max(limit, 1), 1000), rpc_timeout=timedelta(seconds=RPC_TIMEOUT))]

    @staticmethod
    async def _owner(ex: WorkflowExecution) -> int | None:
        try:
            return _memo_user_id(dict(await ex.memo()))
        except Exception:
            return None

    async def _to_run(self, ex: WorkflowExecution) -> WorkflowRun | None:
        """A workflow execution (listed or described) as the panel sees it;
        None when it is not one of ours."""
        kind = KIND_BY_TYPE.get(ex.workflow_type)
        if kind is None:
            return None
        try:
            memo = dict(await ex.memo())
        except Exception:
            memo = {}
        params = memo.get("params")
        status = STATUS_MAP.get(ex.status, "unknown") if ex.status is not None else "unknown"
        run = WorkflowRun(
            id=ex.id, kind=kind, status=status, params=params if isinstance(params, dict) else {},  # type: ignore[arg-type]
            started_at=ex.start_time, closed_at=ex.close_time if status not in ("running", "queued") else None,
            url=self.settings.run_url(ex.id), backend="temporal",
            requested_by=str(memo.get("requested_by") or ""), user_id=_memo_user_id(memo),
            source=str(memo.get("source") or ""), model=str(memo.get("model") or ""), note=str(memo.get("note") or ""),
        )
        if (cached := self._outcomes.get(ex.id)) is not None:
            run.result, run.error = cached
        return run

    async def _progress(self, handle: WorkflowHandle[Any, Any], desc: WorkflowExecutionDescription | None) -> RunProgress | None:
        """The workflow's own stage (a query), refined by the latest heartbeat
        of a running activity. A
        run no worker has picked up has no progress, and is not queried when
        no worker is polling, since the query would only time out."""
        picked_up = desc is not None and (desc.history_length > 2 or len(desc.raw_description.pending_activities) > 0)
        if not picked_up and await self._worker_available() is False:
            return None
        try:
            async with asyncio.timeout(QUERY_TIMEOUT):
                progress: RunProgress = await handle.query(PROGRESS_QUERY, result_type=RunProgress,
                                                           rpc_timeout=timedelta(seconds=QUERY_TIMEOUT))
        except Exception as e:
            log.debug("progress query for %s failed: %s", handle.id, _rpc_message(e))
            return None
        if desc is not None:
            for pending in desc.raw_description.pending_activities:
                if not pending.heartbeat_details.payloads:
                    continue
                try:
                    beat = (await self.client.data_converter.decode(pending.heartbeat_details.payloads, [RunProgress]))[0]
                except Exception:
                    continue
                if isinstance(beat, RunProgress) and (progress.updated_at is None or beat.updated_at is None
                                                      or beat.updated_at >= progress.updated_at):
                    progress = progress.model_copy(update={"message": beat.message, "done": beat.done,
                                                           "total": beat.total, "updated_at": beat.updated_at})
        return progress

    async def _outcome(self, handle: WorkflowHandle[Any, Any], run: WorkflowRun) -> tuple[dict[str, Any] | None, str] | None:
        """A closed run's result or error, fetched once and cached."""
        if (cached := self._outcomes.get(run.id)) is not None:
            return cached
        try:
            async with asyncio.timeout(RPC_TIMEOUT * 2):
                value = await handle.result(rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
            outcome: tuple[dict[str, Any] | None, str] = (_normalize_result(run.kind, value), "")
        except WorkflowFailureError as e:
            outcome = (None, _failure_message(e))
        except Exception as e:
            log.warning("could not read the outcome of %s: %s", run.id, _rpc_message(e))
            return None
        self._remember(run.id, outcome)
        return outcome

    # --- RunService ----------------------------------------------------------------------

    async def start(self, request: RunRequest) -> WorkflowRun:
        kind = request.kind
        now = datetime.now(timezone.utc)
        run_id = new_run_id(kind, now)
        params = request.params.model_dump(mode="json")
        memo = {"kind": kind, "requested_by": request.requested_by, "user_id": request.user_id,
                "source": request.source, "model": request.model, "note": request.note, "params": params}
        try:
            async with asyncio.timeout(RPC_TIMEOUT * 2):
                await self.client.start_workflow(
                    WORKFLOW_TYPES[kind], request, id=run_id, task_queue=self.task_queue, memo=memo,
                    execution_timeout=EXECUTION_TIMEOUTS[kind], static_summary=run_summary(request)[:400],
                    static_details=request.note or None, rpc_timeout=timedelta(seconds=RPC_TIMEOUT),
                )
        except Exception as e:
            log.warning("Temporal refused %s: %s", run_id, _rpc_message(e))
            raise RunsUnavailable(f"Temporal did not accept the run: {_rpc_message(e)}") from e
        log.info("started %s (%s)", run_id, run_summary(request))
        run = WorkflowRun(
            id=run_id, kind=kind, status="running", params=params, started_at=now,
            progress=RunProgress(stage="starting", message="Waiting for a worker to pick it up", updated_at=now),
            url=self.settings.run_url(run_id), backend="temporal", requested_by=request.requested_by,
            user_id=request.user_id, source=request.source, model=request.model, note=request.note,
        )
        record_run_started(run)
        return run

    async def get(self, run_id: str) -> WorkflowRun | None:
        handle = self.client.get_workflow_handle(run_id)
        try:
            async with asyncio.timeout(RPC_TIMEOUT + 1):
                desc = await handle.describe(rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
        except RPCError as e:
            if e.status != RPCStatusCode.NOT_FOUND:
                log.warning("describe %s failed: %s", run_id, _rpc_message(e))
            return None
        except Exception as e:
            log.warning("describe %s failed: %s", run_id, _rpc_message(e))
            return None
        run = await self._to_run(desc)
        if run is None:
            return None
        if run.open:
            run.progress = await self._progress(handle, desc)
        elif (outcome := await self._outcome(handle, run)) is not None:
            run.result, run.error = outcome
        return run

    async def list(self, limit: int = 20, kind: WorkflowKind | None = None, *,
                   user_id: int | None = None) -> list[WorkflowRun]:
        limit = max(1, min(limit, 200))
        try:
            executions = await self._list_executions(self._filter(kind), limit if user_id is None else USER_SCAN)
        except Exception as e:
            log.warning("listing runs failed: %s", _rpc_message(e))
            return []
        runs = [run for ex in executions if (run := await self._to_run(ex)) is not None
                and (user_id is None or run.user_id == user_id)]
        # Visibility lists open runs first, then by close time; show newest first.
        runs.sort(key=lambda r: r.started_at, reverse=True)
        runs = runs[:limit]
        open_runs = [r for r in runs if r.open]
        # A finished run's outcome is immutable, so it is fetched once and kept;
        # the newest few that are not cached yet are fetched here, so the panel
        # shows what a run produced without a detail request per card.
        uncached = [r for r in runs if not r.open and r.id not in self._outcomes][:LIST_OUTCOMES]
        if open_runs or uncached:
            gate = asyncio.Semaphore(LIST_CONCURRENCY)

            async def refresh(run: WorkflowRun) -> None:
                async with gate:
                    handle = self.client.get_workflow_handle(run.id)
                    try:
                        async with asyncio.timeout(RPC_TIMEOUT):
                            desc = await handle.describe(rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
                    except Exception:
                        return
                    run.progress = await self._progress(handle, desc)

            async def settle(run: WorkflowRun) -> None:
                async with gate:
                    outcome = await self._outcome(self.client.get_workflow_handle(run.id), run)
                    if outcome is not None:
                        run.result, run.error = outcome

            await asyncio.gather(*(refresh(r) for r in open_runs), *(settle(r) for r in uncached))
        return runs

    async def _count_by_status(self, kind: str, since_iso: str) -> dict[str, int] | None:
        """Runs of one kind in the window by status, from visibility; None when
        the server cannot group (the caller then counts a sample)."""
        query = (f"WorkflowType = '{WORKFLOW_TYPES[kind]}' AND TaskQueue = '{self.task_queue}' "
                 f"AND StartTime >= '{since_iso}' GROUP BY ExecutionStatus")
        try:
            async with asyncio.timeout(RPC_TIMEOUT + 1):
                counted = await self.client.count_workflows(query, rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
        except Exception as e:
            log.debug("grouped count unavailable: %s", _rpc_message(e))
            return None
        out: dict[str, int] = {}
        for group in counted.groups:
            raw = str(group.group_values[0]) if group.group_values else ""
            status = VISIBILITY_STATUS.get(raw, raw.lower() or "unknown")
            out[status] = out.get(status, 0) + group.count
        if not counted.groups and counted.count:
            return None
        return out

    async def stats(self, days: int = 30, *, user_id: int | None = None) -> RunStats:
        days = max(1, days)
        since = datetime.now(timezone.utc) - timedelta(days=days)
        since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        out = RunStats(window_days=days)

        async def no_counts() -> list[None]:
            return [None] * len(WORKFLOW_KINDS)

        # Grouped visibility counts cannot see the memo's owner, so one
        # account's statistics come from the sample alone.
        sampled, grouped, recent = await asyncio.gather(
            self._list_executions(f"{self._filter()} AND StartTime >= '{since_iso}'", STATS_SAMPLE),
            asyncio.gather(*(self._count_by_status(k, since_iso) for k in WORKFLOW_KINDS))
            if user_id is None else no_counts(),
            self.list(10, user_id=user_id),
            return_exceptions=True,
        )
        if isinstance(sampled, BaseException):
            log.warning("run statistics: listing failed: %s", _rpc_message(sampled))
            sample: list[WorkflowExecution] = []
        else:
            sample = sampled
        if user_id is not None:
            sample = [ex for ex in sample if await self._owner(ex) == user_id]
        if isinstance(grouped, BaseException):
            grouped = [None] * len(WORKFLOW_KINDS)
        if isinstance(recent, BaseException):
            recent = []
        by_kind: dict[str, dict[str, int]] = {}
        durations: dict[str, list[float]] = {}
        last: dict[str, datetime | None] = {}
        for ex in sample:
            kind = KIND_BY_TYPE.get(ex.workflow_type)
            if kind is None:
                continue
            status = STATUS_MAP.get(ex.status, "unknown") if ex.status is not None else "unknown"
            counts = by_kind.setdefault(kind, {})
            counts[status] = counts.get(status, 0) + 1
            current = last.get(kind)
            last[kind] = ex.start_time if current is None or ex.start_time > current else current
            if status == "completed" and ex.close_time is not None:
                durations.setdefault(kind, []).append((ex.close_time - ex.start_time).total_seconds())
        # Exact counts from visibility where the server can group; the sample otherwise.
        for kind, counts in zip(WORKFLOW_KINDS, grouped, strict=True):
            if counts is not None:
                if counts:
                    by_kind[kind] = counts
                else:
                    by_kind.pop(kind, None)
        out.by_kind = by_kind
        out.total = sum(sum(c.values()) for c in by_kind.values())
        out.running = sum(c.get("running", 0) for c in by_kind.values())
        out.failed = sum(c.get("failed", 0) + c.get("timed_out", 0) for c in by_kind.values())
        out.avg_duration_s = {k: round(sum(v) / len(v), 3) for k, v in durations.items() if v}
        out.last_run_at = last
        out.recent = recent
        return out

    async def cancel(self, run_id: str) -> bool:
        handle = self.client.get_workflow_handle(run_id)
        try:
            async with asyncio.timeout(RPC_TIMEOUT * 2):
                desc = await handle.describe(rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
                if KIND_BY_TYPE.get(desc.workflow_type) is None or desc.status != WorkflowExecutionStatus.RUNNING:
                    return False
                await handle.cancel(reason="Cancelled from the R3 Controller",
                                    rpc_timeout=timedelta(seconds=RPC_TIMEOUT))
        except Exception as e:
            log.warning("cancel %s failed: %s", run_id, _rpc_message(e))
            return False
        return True

    def describe(self) -> RunsInfo:
        reason = ""
        if self._worker_seen is False:
            reason = (f"No worker is polling task queue {self.task_queue!r}, so runs wait. Start one with "
                      "`uv run r3-worker`.")
        return RunsInfo(backend="temporal", configured=True, namespace=self.settings.temporal_namespace,
                        task_queue=self.task_queue, ui_url=self.settings.temporal_ui_url, reason=reason)

    async def close(self) -> None:
        return None
