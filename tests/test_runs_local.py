"""The in-process runner: a stats run executes the shared flow to
completion against the stub database, with a typed result and progress;
failures read well, cancellation works, one run per kind executes at a
time, and the builder picks the right runner."""

import asyncio
from datetime import datetime, timezone

import pytest

from r3.runs import service as service_module
from r3.runs.local import LOCAL_REASON, LocalRuns
from r3.runs.models import RunRequest, StatsParams, StatsSnapshot, WorkflowRun
from r3.runs.service import NoRuns, build_run_service
from r3.runs.stats import hide_run_internals
from r3.settings import ControllerSettings
from tests.test_routes import StubDB


class HeldStub(StubDB):
    """A stub whose overview waits for ``release`` while ``hold`` is set, so
    a run can be caught open."""

    def __init__(self) -> None:
        super().__init__()
        self.hold = False
        self.release = asyncio.Event()

    async def overview(self, *, user_id=None):
        if self.hold:
            await self.release.wait()
        return await super().overview(user_id=user_id)


@pytest.fixture
def stub():
    return HeldStub()


@pytest.fixture
async def make_runs(stub):
    made: list[LocalRuns] = []

    def make() -> LocalRuns:
        runs = LocalRuns(stub, retry_delay=0)
        made.append(runs)
        return runs

    yield make
    for runs in made:
        await runs.close()


async def run_to_end(runs: LocalRuns, params, **kw):
    started = await runs.start(RunRequest(params=params, **kw))
    assert started.status == "running" and started.backend == "local" and started.progress.stage == "starting"
    done = await runs.wait(started.id, timeout=10)
    assert done is not None and not done.open, done
    return done


async def test_stats_run_returns_the_snapshot(make_runs):
    run = await run_to_end(make_runs(), StatsParams(days=7), note="numbers", source="chat")
    assert run.status == "completed" and run.closed_at is not None and run.duration_s >= 0
    snapshot = StatsSnapshot.model_validate(run.result)
    assert (snapshot.total_jobs, snapshot.new_jobs, snapshot.window_days) == (100, 40, 7)
    assert snapshot.runs is None
    assert run.progress.stage == "finished" and run.progress.message.startswith("100 jobs, 40 unreviewed")
    assert (run.note, run.source, run.params["days"]) == ("numbers", "chat", 7)


async def test_stats_run_of_an_account_is_that_accounts_numbers(make_runs, stub):
    """R3-09: a signed-in account's stats run must not return the portal-wide snapshot."""
    seen: list[int | None] = []
    original = stub.overview

    async def overview(*, user_id=None):
        seen.append(user_id)
        stats = await original(user_id=user_id)
        return stats.model_copy(update={"total_jobs": 7, "new_jobs": 3}) if user_id else stats

    stub.overview = overview
    run = await run_to_end(make_runs(), StatsParams(days=7), user_id=5, requested_by="dana")
    snapshot = StatsSnapshot.model_validate(run.result)
    assert seen == [5] and (snapshot.total_jobs, snapshot.new_jobs) == (7, 3)
    # The local operator (sign-in off, no account) and schedules keep the portal view.
    seen.clear()
    portal = StatsSnapshot.model_validate((await run_to_end(make_runs(), StatsParams(days=7))).result)
    assert seen == [None] and portal.total_jobs == 100


async def test_a_failing_step_fails_the_run_readably(make_runs, stub):
    async def down(*, user_id=None):
        raise ConnectionError("database down")

    stub.overview = down
    run = await run_to_end(make_runs(), StatsParams())
    assert run.status == "failed" and run.progress.stage == "failed"
    assert run.error.startswith("Statistics failed: ConnectionError: database down")


async def test_cancel_stops_a_running_run(make_runs, stub):
    stub.hold = True
    runs = make_runs()
    started = await runs.start(RunRequest(params=StatsParams()))
    await asyncio.sleep(0.05)
    assert await runs.cancel(started.id)
    run = await runs.get(started.id)
    assert run.status == "cancelled" and run.closed_at is not None
    assert not await runs.cancel(started.id)
    assert not await runs.cancel("r3-stats-nope")
    await asyncio.sleep(0.01)
    assert started.id not in runs._tasks


async def test_one_run_per_kind_at_a_time(make_runs, stub):
    stub.hold = True
    runs = make_runs()
    first = await runs.start(RunRequest(params=StatsParams()))
    second = await runs.start(RunRequest(params=StatsParams()))
    assert (first.status, second.status) == ("running", "queued")
    assert second.progress.stage == "queued"
    stub.release.set()
    assert (await runs.wait(first.id)).status == "completed"
    assert (await runs.wait(second.id)).status == "completed"


async def test_list_stats_and_describe(make_runs, stub):
    runs = make_runs()
    await run_to_end(runs, StatsParams())
    await run_to_end(runs, StatsParams(days=7))

    async def down(*, user_id=None):
        raise ConnectionError("down")

    stub.overview = down
    await run_to_end(runs, StatsParams())
    listed = await runs.list(10)
    assert [r.kind for r in listed] == ["stats", "stats", "stats"]
    assert [r.kind for r in await runs.list(10, kind="stats")] == ["stats"] * 3
    stats = await runs.stats(30)
    assert stats.total == 3 and stats.failed == 1 and stats.running == 0
    assert stats.by_kind == {"stats": {"completed": 2, "failed": 1}}
    assert set(stats.avg_duration_s) == {"stats"} and stats.last_run_at["stats"] == listed[0].started_at
    assert len(stats.recent) == 3
    info = runs.describe()
    assert (info.backend, info.configured, info.reason) == ("local", True, LOCAL_REASON)


async def test_old_runs_are_evicted(stub):
    runs = LocalRuns(stub, retry_delay=0, max_runs=3)
    try:
        ids = [(await run_to_end(runs, StatsParams())).id for _ in range(5)]
        assert [r.id for r in await runs.list(10)] == list(reversed(ids[2:]))
    finally:
        await runs.close()


# --- choosing a runner ----------------------------------------------------------------


async def test_build_run_service_modes(stub, monkeypatch):
    assert isinstance(await build_run_service(ControllerSettings(runs_backend="none"), stub), NoRuns)
    local = await build_run_service(ControllerSettings(), stub)
    assert isinstance(local, LocalRuns) and local.describe().reason == LOCAL_REASON
    missing = await build_run_service(ControllerSettings(runs_backend="temporal"), stub)
    assert isinstance(missing, NoRuns) and "TEMPORAL_ADDRESS" in missing.describe().reason

    from r3.runs import temporal

    async def refuse(cls, settings):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(temporal.TemporalRuns, "connect", classmethod(refuse))
    unreachable = ControllerSettings(temporal_address="temporal.invalid:7233")
    fallback = await build_run_service(unreachable, stub)
    assert isinstance(fallback, LocalRuns)
    assert fallback.describe().reason.startswith("Temporal unreachable at temporal.invalid:7233: ConnectionError")
    strict = await build_run_service(ControllerSettings(temporal_address="temporal.invalid:7233",
                                                        runs_backend="temporal"), stub)
    assert isinstance(strict, NoRuns) and strict.describe().configured is False
    for runs in (local, fallback):
        await runs.close()
    assert service_module.new_run_id("stats").startswith("r3-stats-")


def test_a_regular_account_never_sees_the_temporal_link():
    run = WorkflowRun(id="r", kind="stats", status="completed", started_at=datetime.now(timezone.utc),
                      url="https://cloud.temporal.io/namespaces/x/workflows/r")
    assert hide_run_internals(run).url == ""
    assert hide_run_internals(run.model_copy(update={"url": ""})).url == ""
