"""Runs on a real Temporal server (the CLI dev server, started once for this
module and downloaded on first use): the workflows and activities executed
by the worker R3 ships, observed through ``TemporalRuns``: statuses, memo
fields, typed results, durations, statistics, a failure's message, progress
from the workflow query, cancellation, and a run nobody picks up."""

import asyncio
import time
import uuid

import pytest
import pytest_asyncio
from temporalio.client import WorkflowFailureError
from temporalio.contrib.pydantic import pydantic_data_converter
from temporalio.testing import WorkflowEnvironment

from r3.runs.activities import Activities
from r3.runs.models import RunRequest, StatsParams, StatsSnapshot
from r3.runs.temporal import TemporalRuns
from r3.settings import ControllerSettings
from r3.worker import build_worker
from tests.test_routes import StubDB
from tests.test_runs_local import HeldStub

pytestmark = [pytest.mark.temporal, pytest.mark.asyncio(loop_scope="module")]


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def env():
    try:
        environment = await WorkflowEnvironment.start_local(data_converter=pydantic_data_converter,
                                                            dev_server_log_level="error")
    except Exception as e:  # no binary and no network: an environment problem, not a code one
        pytest.skip(f"Temporal dev server unavailable: {e}")
    yield environment
    await environment.shutdown()


def settings_for(queue: str) -> ControllerSettings:
    return ControllerSettings(temporal_address="localhost:7233", temporal_task_queue=queue,
                              temporal_ui_url="http://localhost:8233/namespaces/default/workflows")


def fresh_queue() -> str:
    return f"r3-test-{uuid.uuid4().hex[:8]}"


async def eventually(check, timeout: float = 10.0):
    """Visibility is eventually consistent: poll until ``check`` passes."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return await check()
        except AssertionError:
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(0.2)


async def test_runs_execute_and_read_back(env):
    queue = fresh_queue()
    settings = settings_for(queue)
    stub = StubDB()
    runs = TemporalRuns(env.client, settings)
    who = dict(requested_by="tester", source="panel", model="openrouter:openai/gpt-4.1-mini")
    async with build_worker(env.client, settings, Activities(stub)):
        started = [await runs.start(RunRequest(params=StatsParams(days=7), note="numbers", **who)),
                   await runs.start(RunRequest(params=StatsParams(days=30), **who))]
        for run in started:
            assert run.status == "running" and run.backend == "temporal" and run.id.startswith("r3-stats-")
            assert run.url == f"http://localhost:8233/namespaces/default/workflows/{run.id}"
            await asyncio.wait_for(env.client.get_workflow_handle(run.id).result(), 30)

    first = await runs.get(started[0].id)
    assert first.status == "completed" and first.closed_at and first.duration_s > 0 and first.progress is None
    assert (first.requested_by, first.source, first.model, first.note) == (
        "tester", "panel", "openrouter:openai/gpt-4.1-mini", "numbers")
    assert first.params["days"] == 7 and first.url.endswith(first.id)
    snapshot = StatsSnapshot.model_validate(first.result)
    assert snapshot.window_days == 7 and snapshot.total_jobs == 100

    async def listed():
        rows = await runs.list(10)
        assert len(rows) == 2 and all(r.status == "completed" for r in rows), rows
        return rows

    rows = await eventually(listed)
    assert [r.started_at for r in rows] == sorted((r.started_at for r in rows), reverse=True)
    assert all(r.result is not None for r in rows)  # outcomes are attached to the listing
    assert len(await runs.list(10, kind="stats")) == 2

    async def counted():
        s = await runs.stats(30)
        assert s.total == 2, s
        return s

    run_stats = await eventually(counted)
    assert run_stats.by_kind == {"stats": {"completed": 2}}
    assert (run_stats.running, run_stats.failed) == (0, 0)
    assert set(run_stats.avg_duration_s) == {"stats"} and run_stats.avg_duration_s["stats"] > 0
    assert len(run_stats.recent) == 2

    info = runs.describe()
    assert (info.backend, info.configured, info.task_queue, info.namespace) == ("temporal", True, queue, "default")


async def test_a_failed_run_carries_its_message(env):
    queue = fresh_queue()
    settings = settings_for(queue)
    stub = StubDB()

    async def down(*, user_id=None):
        raise ConnectionError("database down")

    stub.overview = down
    runs = TemporalRuns(env.client, settings)
    async with build_worker(env.client, settings, Activities(stub)):
        run = await runs.start(RunRequest(params=StatsParams()))
        with pytest.raises(WorkflowFailureError):
            await asyncio.wait_for(env.client.get_workflow_handle(run.id).result(), 60)
    failed = await runs.get(run.id)
    assert failed.status == "failed" and failed.result is None
    assert failed.error.startswith("Statistics failed:") and "database down" in failed.error

    async def counted():
        s = await runs.stats(1)
        assert s.failed == 1 and s.by_kind == {"stats": {"failed": 1}}, s

    await eventually(counted)


async def test_progress_comes_from_the_workflow_and_cancel_ends_the_run(env):
    queue = fresh_queue()
    settings = settings_for(queue)
    stub = HeldStub()
    stub.hold = True
    runs = TemporalRuns(env.client, settings)
    async with build_worker(env.client, settings, Activities(stub)):
        run = await runs.start(RunRequest(params=StatsParams(days=7)))

        async def reporting():
            live = await runs.get(run.id)
            assert live.status == "running" and live.progress is not None, live
            assert live.progress.stage == "computing", live
            return live

        live = await eventually(reporting)
        assert live.progress.message == "Computing statistics over 7 days"
        listed = (await runs.list(5))[0]
        assert listed.id == run.id and listed.progress is not None

        assert await runs.cancel(run.id)

        async def cancelled():
            closed = await runs.get(run.id)
            assert closed.status == "cancelled", closed
            return closed

        closed = await eventually(cancelled)
        assert closed.error == "Cancelled" and not await runs.cancel(run.id)
        stub.release.set()


async def test_a_run_without_a_worker_stays_running_without_progress(env):
    queue = fresh_queue()
    runs = TemporalRuns(env.client, settings_for(queue))
    run = await runs.start(RunRequest(params=StatsParams()))
    began = time.monotonic()
    waiting = await runs.get(run.id)
    assert time.monotonic() - began < 8
    assert waiting.status == "running" and waiting.progress is None and waiting.error == ""
    assert "No worker is polling task queue" in runs.describe().reason
    assert await runs.get("r3-stats-does-not-exist") is None
    await env.client.get_workflow_handle(run.id).terminate(reason="test cleanup")
    terminated = await runs.get(run.id)
    assert terminated.status == "terminated" and terminated.error == "Terminated: test cleanup"


async def test_the_worker_command_executes_runs_and_stops_cleanly(env, monkeypatch):
    from r3 import worker as worker_command
    from r3.db import Database

    queue = fresh_queue()
    settings = ControllerSettings(temporal_address=env.client.service_client.config.target_host,
                                 temporal_task_queue=queue)
    stub = StubDB()
    closed: list[bool] = []

    async def close():
        closed.append(True)

    async def open_database(cls, dsn):
        return stub

    async def ensure_events_table():
        return None

    stub.close = close
    stub.ensure_events_table = ensure_events_table
    monkeypatch.setattr(Database, "open", classmethod(open_database))
    stop = asyncio.Event()
    worker = asyncio.create_task(worker_command.run_worker(settings, "postgresql://unused", stop))
    runs = TemporalRuns(env.client, settings)
    run = await runs.start(RunRequest(params=StatsParams(days=7)))
    result = await asyncio.wait_for(env.client.get_workflow_handle(run.id).result(), 30)
    stop.set()
    await asyncio.wait_for(worker, 30)
    assert StatsSnapshot.model_validate(result).total_jobs == 100 and closed == [True]
