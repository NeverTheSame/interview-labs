"""Fakes shared by the route, agent, runs and UI tests: an in-memory run
service, an advisor that replays scripted events, and a job-overview
summarizer."""

from __future__ import annotations

import itertools
from datetime import datetime, timezone
from typing import AsyncIterator, Iterable

from r3.agent.contracts import ChatTurnRequest
from r3.agent.events import DoneEvent
from r3.job_summary import Overview
from r3.runs.models import RunRequest, RunsInfo, RunStats, WorkflowKind, WorkflowRun
from r3.runs.service import RunsUnavailable
from r3.settings import ControllerSettings


class FakeRuns:
    """A ``RunService`` that records requests and completes runs on demand."""

    def __init__(self, *, configured: bool = True, backend: str = "local", reason: str = "") -> None:
        self.configured = configured
        self.backend = backend
        self.reason = reason
        self.runs: dict[str, WorkflowRun] = {}
        self.requests: list[RunRequest] = []
        self._ids = itertools.count(1)

    async def start(self, request: RunRequest) -> WorkflowRun:
        if not self.configured:
            raise RunsUnavailable(self.reason or "runs are not configured")
        self.requests.append(request)
        run = WorkflowRun(
            id=f"r3-{request.kind}-test-{next(self._ids):04d}", kind=request.kind, status="running",
            params=request.params.model_dump(mode="json"), started_at=datetime.now(timezone.utc),
            backend=self.backend, requested_by=request.requested_by, source=request.source,  # type: ignore[arg-type]
            model=request.model, note=request.note, user_id=request.user_id,
        )
        self.runs[run.id] = run
        return run

    def complete(self, run_id: str, result: dict | None = None, *, status: str = "completed", error: str = "") -> WorkflowRun:
        run = self.runs[run_id].model_copy(update={"status": status, "closed_at": datetime.now(timezone.utc),
                                                   "result": result, "error": error})
        self.runs[run_id] = run
        return run

    async def get(self, run_id: str) -> WorkflowRun | None:
        return self.runs.get(run_id)

    async def list(self, limit: int = 20, kind: WorkflowKind | None = None, *,
                   user_id: int | None = None) -> list[WorkflowRun]:
        rows = [r for r in self.runs.values()
                if (kind is None or r.kind == kind) and (user_id is None or r.user_id == user_id)]
        return sorted(rows, key=lambda r: r.started_at, reverse=True)[:limit]

    async def stats(self, days: int = 30, *, user_id: int | None = None) -> RunStats:
        by_kind: dict[str, dict[str, int]] = {}
        runs = [r for r in self.runs.values() if user_id is None or r.user_id == user_id]
        for r in runs:
            by_kind.setdefault(r.kind, {})
            by_kind[r.kind][r.status] = by_kind[r.kind].get(r.status, 0) + 1
        return RunStats(window_days=days, total=len(runs), running=sum(r.open for r in runs),
                        failed=sum(r.status == "failed" for r in runs), by_kind=by_kind,
                        recent=await self.list(10, user_id=user_id))

    async def cancel(self, run_id: str) -> bool:
        if run_id in self.runs and self.runs[run_id].open:
            self.complete(run_id, status="cancelled")
            return True
        return False

    def describe(self) -> RunsInfo:
        return RunsInfo(backend=self.backend if self.configured else "none", configured=self.configured,  # type: ignore[arg-type]
                        namespace="test", task_queue="r3-test", reason=self.reason)

    async def close(self) -> None:
        return None


class ScriptedAdvisor:
    """An ``AdvisorService`` that streams a fixed list of events per turn and
    records what it was asked."""

    def __init__(self, settings: ControllerSettings, events: Iterable[object] = ()) -> None:
        self.settings = settings
        self.events = list(events)
        self.turns: list[ChatTurnRequest] = []
        self.users: list[object] = []  # the account each turn was asked by

    async def stream_turn(self, turn: ChatTurnRequest, database, runs, *, user=None) -> AsyncIterator[object]:
        self.turns.append(turn)
        self.users.append(user)
        for event in self.events:
            yield event
        if not self.events or not isinstance(self.events[-1], DoneEvent):
            yield DoneEvent()


def test_settings(**overrides) -> ControllerSettings:
    """Settings with every provider key present, so model choice is not the
    thing under test."""
    base = dict(api_keys={"openrouter": "or-test", "openai": "oa-test", "typesafe": "ts-test"},
                default_model="openrouter:openrouter/free")
    base.update(overrides)
    return ControllerSettings(**base)


class FakeSummarizer:
    """A job-overview summarizer that answers at once with a fixed text and
    counts its calls. ``engine="llm"`` makes the app cache what it says."""

    def __init__(self, text: str = "Acme AI trains models for logistics. The role ships them to production on "
                                   "Kubernetes.", engine: str = "llm") -> None:
        self.text, self.engine, self.calls = text, engine, 0

    def uses_model(self) -> bool:
        return self.engine == "llm"

    async def overview(self, job) -> Overview:
        self.calls += 1
        return Overview(text=self.text, engine=self.engine, model="fake-model" if self.engine == "llm" else "")
