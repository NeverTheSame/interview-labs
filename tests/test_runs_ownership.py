"""Runs belong to accounts: the owner round-trips through every runner,
regular accounts see and control only their own runs, and the panel and
stats hide operator details from regular accounts."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from temporalio.client import WorkflowExecutionStatus

from r3.runs.models import RunRequest, StatsParams
from r3.runs.temporal import WORKFLOW_TYPES, TemporalRuns
from r3.web.auth import User, current_user
from tests import fakes
from tests.test_routes import StubDB
from tests.test_runs_local import make_runs, run_to_end, stub  # noqa: F401  (fixtures)
from tests.test_runs_routes import make_client

ADMIN = User(id=1, account_name="admin", is_admin=True)
ALICE = User(id=2, account_name="alice", is_admin=False)
BOB = User(id=3, account_name="bob", is_admin=False)


# --- the runners ----------------------------------------------------------------------


async def test_local_runs_keep_their_owner_and_filter_by_it(make_runs):  # noqa: F811
    runs = make_runs()
    mine = await run_to_end(runs, StatsParams(), user_id=7)
    await run_to_end(runs, StatsParams(), user_id=8)
    await run_to_end(runs, StatsParams())
    assert mine.user_id == 7 and (await runs.get(mine.id)).user_id == 7
    assert [r.id for r in await runs.list(user_id=7)] == [mine.id]
    assert len(await runs.list()) == 3
    stats = await runs.stats(user_id=7)
    assert stats.total == 1 and [r.id for r in stats.recent] == [mine.id]
    assert (await runs.stats()).total == 3


async def test_fake_runs_keep_their_owner_and_filter_by_it():
    runs = fakes.FakeRuns()
    mine = await runs.start(RunRequest(params=StatsParams(), user_id=7))
    await runs.start(RunRequest(params=StatsParams()))
    assert mine.user_id == 7
    assert [r.id for r in await runs.list(user_id=7)] == [mine.id]
    assert (await runs.stats(user_id=7)).total == 1 and (await runs.stats()).total == 2


class FakeExecution:
    def __init__(self, run_id: str, kind: str, memo: dict) -> None:
        self.id = run_id
        self.workflow_type = WORKFLOW_TYPES[kind]
        self.status = WorkflowExecutionStatus.RUNNING
        self.start_time = datetime.now(timezone.utc)
        self.close_time = None
        self._memo = memo

    async def memo(self) -> dict:
        return self._memo


class FakeHandle:
    async def describe(self, **_):
        raise RuntimeError("no server")


class FakeTemporalClient:
    """Records start_workflow's memo and lists what was started."""

    def __init__(self) -> None:
        self.executions: list[FakeExecution] = []

    async def start_workflow(self, workflow, arg, *, id, memo, **_):
        self.executions.append(FakeExecution(id, arg.kind, memo))

    async def list_workflows(self, query, **_):
        for ex in list(self.executions):
            yield ex

    def get_workflow_handle(self, run_id):
        return FakeHandle()


async def test_temporal_memo_carries_the_owner_both_ways():
    client = FakeTemporalClient()
    runs = TemporalRuns(client, fakes.test_settings())  # type: ignore[arg-type]
    started = await runs.start(RunRequest(params=StatsParams(), user_id=7, requested_by="alice"))
    await runs.start(RunRequest(params=StatsParams()))
    assert started.user_id == 7
    assert client.executions[0]._memo["user_id"] == 7 and client.executions[1]._memo["user_id"] is None
    client.executions.append(FakeExecution("r3-stats-old", "stats", {"requested_by": "controller"}))  # pre-accounts
    listed = {r.id: r.user_id for r in await runs.list()}
    assert listed == {started.id: 7, client.executions[1].id: None, "r3-stats-old": None}
    assert [r.id for r in await runs.list(user_id=7)] == [started.id]
    stats = await runs.stats(user_id=7)
    assert stats.total == 1 and [r.id for r in stats.recent] == [started.id]


# --- the HTTP API ---------------------------------------------------------------------


@pytest.fixture
async def api():
    runs = fakes.FakeRuns()
    async with make_client(StubDB(), runs) as c:
        app = c._transport.app  # type: ignore[attr-defined]

        def as_user(user: User) -> None:
            app.dependency_overrides[current_user] = lambda: user

        yield c, runs, as_user
        app.dependency_overrides.clear()


async def test_accounts_see_and_cancel_only_their_own_runs(api):
    c, runs, as_user = api
    as_user(ALICE)
    a = (await c.post("/api/runs", json={"params": {"kind": "stats"}, "user_id": 99, "requested_by": "x"})).json()
    as_user(BOB)
    b = (await c.post("/api/runs", json={"params": {"kind": "stats"}})).json()
    assert runs.requests[0].user_id == ALICE.id and runs.requests[0].requested_by == "alice"
    assert (a["user_id"], b["user_id"]) == (ALICE.id, BOB.id)
    assert [r["id"] for r in (await c.get("/api/runs")).json()] == [b["id"]]
    assert (await c.get(f"/api/runs/{a['id']}")).status_code == 404
    assert (await c.post(f"/api/runs/{a['id']}/cancel")).status_code == 404
    assert (await c.get(f"/api/runs/{b['id']}")).status_code == 200
    as_user(ADMIN)
    assert {r["id"] for r in (await c.get("/api/runs")).json()} == {a["id"], b["id"]}
    assert (await c.get(f"/api/runs/{a['id']}")).status_code == 200
    assert (await c.post(f"/api/runs/{b['id']}/cancel")).status_code == 200


async def test_ownerless_runs_are_hidden_from_accounts(api):
    c, runs, as_user = api
    ownerless = await runs.start(RunRequest(params=StatsParams()))
    as_user(ALICE)
    assert (await c.get(f"/api/runs/{ownerless.id}")).status_code == 404
    assert (await c.get("/api/runs")).json() == []


async def test_panel_and_stats_by_role(api):
    c, runs, as_user = api
    await runs.start(RunRequest(params=StatsParams()))
    as_user(ALICE)
    await c.post("/api/runs", json={"params": {"kind": "stats"}})
    panel = (await c.get("/api/panel")).json()
    assert [r["user_id"] for r in panel["recent_runs"]] == [ALICE.id]
    assert all(r["url"] == "" for r in panel["recent_runs"])
    info = panel["runs_info"]
    assert (info["namespace"], info["task_queue"], info["ui_url"], info["reason"]) == ("", "", "", "")
    assert panel["stats"]["runs"]["total"] == 1
    assert not any(f["label"] == "jobs stored" for items in panel["facts"].values() for f in items)
    assert "connect-temporal" not in {s["id"] for s in panel["suggestions"]}
    stats = (await c.get("/api/stats")).json()
    assert stats["runs"]["total"] == 1
    as_user(ADMIN)
    panel = (await c.get("/api/panel")).json()
    assert len(panel["recent_runs"]) == 2
    assert any(f["label"] == "jobs stored" for items in panel["facts"].values() for f in items)
    stats = (await c.get("/api/stats")).json()
    assert stats["total_jobs"] == 100 and stats["runs"]["total"] == 2
