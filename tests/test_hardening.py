"""Regression tests for the black-box API findings (API-01 ... API-18)."""

from __future__ import annotations

from types import SimpleNamespace

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from r3 import db
from r3.db.jobs import JobQueries
from r3.web import create_app
from r3.web.routes_label import label_filter, parse_seen

from .test_routes import StubDB

BIG = "9223372036854775808"  # int64 max + 1


class FakePool:
    """Records the SQL and arguments a query method sends."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return "UPDATE 1"

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return []


@pytest.fixture
def stub():
    return StubDB()


@pytest.fixture
async def client(stub):
    app = create_app(database=stub)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# API-01
async def test_job_id_above_int64_is_rejected(client, stub):
    assert (await client.get(f"/jobs/{BIG}")).status_code == 422
    assert (await client.post(f"/jobs/{BIG}/status", data={"status": "new"})).status_code == 422
    assert (await client.post(f"/jobs/{BIG}/notes", data={"notes": "x"})).status_code == 422
    assert stub.writes == []


async def test_label_and_resume_ids_above_int64_are_rejected(client):
    for method, path in [("get", f"/label/{BIG}"), ("post", f"/label/{BIG}"),
                         ("get", f"/resume/{BIG}"), ("post", f"/resume/{BIG}"),
                         ("post", f"/resume/{BIG}/primary"), ("post", f"/resume/{BIG}/delete"),
                         ("get", f"/resume/{BIG}/download"), ("get", f"/resume/{BIG}/pdf")]:
        assert (await getattr(client, method)(path)).status_code == 422, path


async def test_page_size_is_clamped(client, stub):
    await client.get("/jobs?page=99999999999999999999&size=99999999")
    assert stub.last_page == (100_000, 50)
    await client.get("/api/jobs?page=99999999999999999999")
    assert stub.last_page[0] == 100_000


# API-02
async def test_days_and_salary_are_clamped(client, stub):
    for headers in ({}, {"HX-Request": "true"}):
        await client.get("/jobs?days=99999999&min_salary=99999999999999999999", headers=headers)
        assert (stub.last_filter.days, stub.last_filter.min_salary) == (3650, 10_000_000)
    await client.get("/api/jobs?days=99999999&min_salary=99999999999999999999")
    assert (stub.last_filter.days, stub.last_filter.min_salary) == (3650, 10_000_000)
    assert label_filter({"days": "99999999"}).days == 3650


# API-05
async def test_api_jobs_bad_value_is_400_not_503(client, stub):
    async def boom(*a, **k):
        raise asyncpg.DataError("value out of range")

    stub.list_jobs = boom
    assert (await client.get("/api/jobs")).status_code == 400


# API-06
async def test_control_characters_are_stripped(client, stub):
    await client.get("/jobs?q=a%00b")
    assert stub.last_filter.search == "ab"
    await client.get("/api/jobs?q=a%00b")
    assert stub.last_filter.search == "ab"
    await client.post("/jobs/1/notes", data={"notes": "a\x00b\tc\nd"})
    assert stub.writes[-1][-1] == "ab\tc\nd"


# API-07
def test_parse_seen_drops_out_of_range_ids():
    assert parse_seen(f"99999999999999999999,{BIG},7") == [7]


# API-10 / API-11
def test_job_search_escapes_like_and_is_capped():
    clause, args = db.build_job_where(db.JobFilter(search="50%_\\x"))
    assert "%50\\%\\_\\\\x%" in args
    assert "ESCAPE" in clause
    _, args = db.build_job_where(db.JobFilter(search="y" * 16000))
    assert len(args[0]) == 202


# API-12
async def test_notes_over_cap_are_rejected_before_any_write():
    pool = FakePool()
    obj = SimpleNamespace(pool=pool)
    with pytest.raises(ValueError, match="10000"):
        await JobQueries.set_user_job_notes(obj, 1, 1, "x" * 10_001)
    assert pool.calls == []


async def test_notes_over_cap_answer_400(client, stub):
    async def too_long(*a):
        raise ValueError("notes are limited to 10000 characters")

    stub.set_user_job_notes = too_long
    r = await client.post("/jobs/1/notes", data={"notes": "x"})
    assert r.status_code == 400 and "10000" in r.text


# API-13
async def test_writes_to_missing_rows_are_404(client, stub):
    async def missing(*a):
        raise LookupError("not found")

    stub.set_user_job_status = stub.set_user_job_notes = missing
    assert (await client.post("/jobs/5/status", data={"status": "new"})).status_code == 404
    assert (await client.post("/jobs/5/notes", data={"notes": "x"})).status_code == 404


# API-15
async def test_healthz_answers_head(client):
    assert (await client.head("/healthz")).status_code == 200


# API-17
async def test_validation_error_input_is_clipped(client):
    r = await client.get("/jobs/" + "a" * 5000)
    assert r.status_code == 422
    assert len(r.text) < 1000
    assert r.json()["detail"][0]["loc"] == ["path", "job_id"]


# API-18
async def test_page_errors_are_html_for_browsers(client):
    html = {"Accept": "text/html,application/xhtml+xml"}
    r = await client.get("/jobs/999", headers=html)
    assert r.status_code == 404 and "text/html" in r.headers["content-type"] and "404" in r.text
    r = await client.get("/jobs/0", headers=html)
    assert r.status_code == 422 and "text/html" in r.headers["content-type"]
    assert (await client.get("/jobs/999")).headers["content-type"] == "application/json"
    r = await client.get("/jobs/999", headers={**html, "HX-Request": "true"})
    assert r.headers["content-type"] == "application/json"
