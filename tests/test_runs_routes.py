"""The runs API end to end over the app, with a fake run service and the
stub database: starting (202, 422, 503), listing, reading (404), cancelling,
statistics, the panel's shape and the JSON job list's query parsing."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient

from r3 import db
from r3.web import create_app
from tests import fakes
from tests.test_routes import JOB, StubDB


@pytest.fixture
def stub():
    return StubDB()


@pytest.fixture
def runs():
    return fakes.FakeRuns()


@asynccontextmanager
async def make_client(stub, runs):
    settings = fakes.test_settings()
    app = create_app(database=stub, settings=settings, runs=runs, advisor=fakes.ScriptedAdvisor(settings))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest.fixture
async def client(stub, runs):
    async with make_client(stub, runs) as c:
        yield c


async def test_start_run_is_accepted(client, runs):
    body = {"params": {"kind": "stats", "days": 7}, "source": "panel", "note": "go"}
    r = await client.post("/api/runs", json=body)
    assert r.status_code == 202, r.text
    run = r.json()
    assert run["kind"] == "stats" and run["status"] == "running" and run["params"]["days"] == 7
    assert runs.requests[0].params.days == 7 and runs.requests[0].source == "panel"


@pytest.mark.parametrize("body", [
    {"params": {"kind": "teleport"}},
    {"params": {"kind": "scout"}},
    {"params": {"kind": "stats", "days": 0}},
    {"params": {"kind": "stats"}, "surprise": 1},
    {},
])
async def test_start_run_rejects_bad_requests(client, body, runs):
    assert (await client.post("/api/runs", json=body)).status_code == 422
    assert runs.requests == []


async def test_start_run_when_runs_are_unavailable():
    async with make_client(StubDB(), fakes.FakeRuns(configured=False, reason="Temporal unreachable")) as c:
        r = await c.post("/api/runs", json={"params": {"kind": "stats"}})
    assert r.status_code == 503 and r.json() == {"detail": "Temporal unreachable"}


async def test_list_get_and_cancel(client):
    ids = []
    for days in (30, 7, 1):
        ids.append((await client.post("/api/runs", json={"params": {"kind": "stats", "days": days}})).json()["id"])
    listed = (await client.get("/api/runs?limit=20&kind=")).json()
    assert [r["id"] for r in listed] == list(reversed(ids))
    assert [r["kind"] for r in (await client.get("/api/runs?kind=stats")).json()] == ["stats"] * 3
    assert len((await client.get("/api/runs?limit=1")).json()) == 1
    assert (await client.get("/api/runs?kind=bogus")).status_code == 422
    assert (await client.get("/api/runs?limit=0")).status_code == 422

    one = await client.get(f"/api/runs/{ids[0]}")
    assert one.status_code == 200 and one.json()["params"]["days"] == 30
    assert (await client.get("/api/runs/r3-nope")).status_code == 404

    assert (await client.post(f"/api/runs/{ids[1]}/cancel")).json() == {"cancelled": True}
    assert (await client.post(f"/api/runs/{ids[1]}/cancel")).json() == {"cancelled": False}
    assert (await client.get(f"/api/runs/{ids[1]}")).json()["status"] == "cancelled"


async def test_stats_includes_run_statistics(client):
    await client.post("/api/runs", json={"params": {"kind": "stats"}})
    r = await client.get("/api/stats?days=7")
    assert r.status_code == 200
    snap = r.json()
    assert (snap["total_jobs"], snap["new_jobs"], snap["window_days"]) == (100, 40, 7)
    assert snap["runs"]["total"] == 1 and snap["runs"]["by_kind"] == {"stats": {"running": 1}}
    assert (await client.get("/api/stats?days=0")).status_code == 422


async def test_stats_when_the_database_is_down(client, stub):
    async def down():
        raise ConnectionError("down")

    stub.overview = down
    r = await client.get("/api/stats")
    assert r.status_code == 503 and "database" in r.json()["detail"]


async def test_panel_shape(client):
    await client.post("/api/runs", json={"params": {"kind": "stats"}})
    r = await client.get("/api/panel")
    assert r.status_code == 200
    panel = r.json()
    assert set(panel) == {"runs_info", "facts", "suggestions", "recent_runs", "stats"}
    assert panel["runs_info"]["backend"] == "local" and panel["runs_info"]["configured"] is True
    assert set(panel["facts"]) == {"jobs", "tracking"}
    assert panel["facts"]["jobs"][0] == {"value": "40", "label": "unreviewed", "warn": False}
    assert [s["id"] for s in panel["suggestions"]] == ["review-new", "top3"]
    assert [r["kind"] for r in panel["recent_runs"]] == ["stats"]
    assert panel["stats"]["total_jobs"] == 100 and panel["stats"]["runs"]["total"] == 1


async def test_panel_survives_a_database_outage(client, stub):
    async def down():
        raise ConnectionError("down")

    stub.job_facets = down
    panel = (await client.get("/api/panel")).json()
    assert panel["facts"] == {} and panel["stats"] is None
    assert panel["runs_info"]["configured"] is True
    assert [s["id"] for s in panel["suggestions"]] == ["stats-today"]


async def test_jobs_api_parses_the_jobs_page_query(client, stub):
    r = await client.get("/api/jobs?q=kafka&status=new&role=DevOps&role=SRE&remote=1&min_salary=200000"
                         "&days=7&page=3&size=99&has_salary=1&employment_family=contract&site=https://x.example/")
    assert r.status_code == 200
    f = stub.last_filter
    assert (f.search, f.status, f.role_categories, f.role_category) == ("kafka", "new", ["DevOps", "SRE"], "DevOps")
    assert f.remote and f.has_salary and (f.min_salary, f.days) == (200000, 7)
    assert (f.employment_family, f.site_url) == ("contract", "https://x.example/")
    assert stub.last_page == (3, 50)  # pages hold at most 50
    body = r.json()
    assert body["total"] == 101 and body["filter_query"] == f.query_string()
    assert (body["page"], body["pages"]) == (2, 3)
    card = body["jobs"][0]
    assert (card["id"], card["position"], card["site_name"]) == (JOB.id, JOB.position, JOB.site_name)


async def test_jobs_api_defaults(client, stub):
    body = (await client.get("/api/jobs")).json()
    assert stub.last_page == (1, 20) and stub.last_filter == db.JobFilter()
    assert body["filter_query"] == "?"
    await client.get("/api/jobs?status=new&limit=10")
    assert stub.last_page == (1, 10) and stub.last_filter.status == "new"


async def test_runs_execute_in_process_through_the_api(stub):
    """No run service injected and no Temporal configured: the app builds the
    in-process runner, and a run started over HTTP completes."""
    settings = fakes.test_settings()
    app = create_app(database=stub, settings=settings, advisor=fakes.ScriptedAdvisor(settings))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            assert (await c.get("/api/panel")).json()["runs_info"]["backend"] == "local"
            started = (await c.post("/api/runs", json={"params": {"kind": "stats", "days": 7}})).json()
            for _ in range(100):
                run = (await c.get(f"/api/runs/{started['id']}")).json()
                if run["status"] not in ("queued", "running"):
                    break
                await asyncio.sleep(0.02)
    assert run["status"] == "completed" and run["backend"] == "local"
    assert run["result"]["total_jobs"] == 100 and run["result"]["window_days"] == 7
