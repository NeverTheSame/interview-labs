"""Every request and every click becomes an audit event (r3/web/audit.py,
r3/web/routes_events.py, static/track.js), tied together by one trace id."""

from __future__ import annotations

import httpx
import pytest

from r3.events import default_log
from r3.web import create_app
from r3.web.auth import SESSION_COOKIE
from tests.test_routes import JOB
from tests.test_user_shell import ShellStub

TRACE = "f" * 32


class AuditStub(ShellStub):
    """ShellStub that keeps the audit events R3 writes."""

    def __init__(self) -> None:
        super().__init__()
        self.events = []

    async def insert_events(self, events):
        self.events.extend(events)


@pytest.fixture
def stub():
    return AuditStub()


async def _call(stub, method, path, *, token="seeker", headers=None, app_kwargs=None, cookies=None, **kwargs):
    app = create_app(database=stub, auth=True, **(app_kwargs or {}))
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        if token:
            c.cookies.set(SESSION_COOKIE, token)
        for k, v in (cookies or {}).items():
            c.cookies.set(k, v)
        return await c.request(method, path, headers={"Origin": "http://test", **(headers or {})}, **kwargs)


def _kinds(stub):
    return [e.kind for e in stub.events]


async def test_a_post_is_recorded_with_the_account_the_route_and_its_subject(stub):
    r = await _call(stub, "POST", f"/jobs/{JOB.id}/status", data={"status": "applied"},
                    headers={"X-R3-Trace": TRACE})
    assert r.status_code == 200
    assert r.headers["x-r3-trace"] == TRACE
    post = next(e for e in stub.events if e.kind == "http.post")
    assert (post.user_id, post.actor, post.trace_id, post.subject) == (42, "user", TRACE, f"job:{JOB.id}")
    assert post.data["route"] == "/jobs/{job_id}/status" and post.data["status"] == 200
    assert "duration_ms" in post.data


async def test_a_page_view_is_recorded_but_polling_and_static_files_are_not(stub):
    await _call(stub, "GET", "/controller", headers={"Accept": "text/html"})
    await _call(stub, "GET", "/api/runs?limit=20")
    await _call(stub, "GET", "/static/app.css")
    await _call(stub, "GET", "/healthz", token=None)
    views = [e for e in stub.events if e.kind == "page.view"]
    assert [v.data["path"] for v in views] == ["/controller"]
    assert all(e.kind in ("page.view",) for e in stub.events), _kinds(stub)


async def test_a_bad_trace_header_is_replaced_by_a_fresh_one(stub):
    r = await _call(stub, "POST", f"/jobs/{JOB.id}/status", data={"status": "applied"},
                    headers={"X-R3-Trace": "not-hex"})
    tid = r.headers["x-r3-trace"]
    assert len(tid) == 32 and tid != "not-hex"


async def test_a_signed_out_post_is_not_attributed_to_anyone(stub):
    r = await _call(stub, "POST", "/login", token=None, data={"account": "dana", "password": "hunter22"})
    assert r.status_code in (200, 303, 400, 401)
    post = next(e for e in stub.events if e.kind == "http.post")
    assert post.user_id is None and post.actor == "system"
    assert "hunter22" not in repr(stub.events)


# --- clicks from the browser -------------------------------------------------------


def _click(label="Upload resume", **over):
    return {"type": "click", "label": label, "target": "button#upload", "path": "/controller", "href": "",
            "trace": TRACE, **over}


async def test_clicks_from_the_page_are_recorded_as_the_account(stub):
    r = await _call(stub, "POST", "/api/events", json={"events": [
        _click(), _click("Jobs", target="a", href="/jobs", trace=""),
        {"type": "view", "label": "Assistant · R3", "path": "/controller"},
        {"type": "change", "label": "Model: free router", "target": "select#model", "path": "/controller"}]})
    assert r.status_code == 204
    ui = [e for e in stub.events if e.kind.startswith("ui.")]
    assert [e.kind for e in ui] == ["ui.click", "ui.click", "ui.view", "ui.change"]
    first = ui[0]
    assert (first.user_id, first.actor, first.trace_id) == (42, "user", TRACE)
    assert first.summary == "Clicked “Upload resume” on /controller"
    assert len(ui[1].trace_id) == 32  # no trace from the page: the request's own
    assert not any(e.kind == "http.post" for e in stub.events)  # the beacon itself is not an action


async def test_a_click_on_a_job_page_is_about_that_job(stub):
    await _call(stub, "POST", "/api/events", json={"events": [_click("I applied", path=f"/jobs/{JOB.id}")]})
    assert stub.events[0].subject == f"job:{JOB.id}"


async def test_click_batches_are_validated_and_bounded(stub):
    too_many = {"events": [_click()] * 51}
    assert (await _call(stub, "POST", "/api/events", json=too_many)).status_code == 422
    bad_type = {"events": [_click(type="keystroke")]}
    assert (await _call(stub, "POST", "/api/events", json=bad_type)).status_code == 422
    long = {"events": [_click("x" * 500)]}
    assert (await _call(stub, "POST", "/api/events", json=long)).status_code == 204
    assert len(stub.events[-1].data["label"]) <= 120


async def test_clicks_need_a_signed_in_account(stub):
    r = await _call(stub, "POST", "/api/events", token=None, json={"events": [_click()]})
    assert r.status_code == 401
    assert stub.events == [] or all(not e.kind.startswith("ui.") for e in stub.events)


async def test_the_event_log_is_detached_when_the_portal_stops(stub):
    await _call(stub, "GET", "/controller", headers={"Accept": "text/html"})
    assert default_log.store is None


async def test_every_page_loads_the_click_tracker(stub):
    r = await _call(stub, "GET", "/controller", headers={"Accept": "text/html"})
    assert "/static/track.js" in r.text


# --- seeing the trail -----------------------------------------------------------------


def _row(i, kind, summary, *, user_id=42, actor="user", subject=f"job:{JOB.id}", trace=TRACE, data=None):
    from datetime import datetime, timezone

    from r3.db import EventRow

    return EventRow(id=i, at=datetime(2026, 10, 11, 15, 7, i, tzinfo=timezone.utc), trace_id=trace, user_id=user_id,
                    actor=actor, kind=kind, subject=subject, summary=summary, data=data or {})


class TrailStub(AuditStub):
    def __init__(self) -> None:
        super().__init__()
        self.rows = []
        self.queries = []

    async def list_events(self, **filters):
        self.queries.append(filters)
        rows = self.rows
        if filters.get("user_id") is not None:
            rows = [r for r in rows if r.user_id == filters["user_id"]]
        if filters.get("subject"):
            rows = [r for r in rows if r.subject == filters["subject"]]
        if filters.get("trace_id"):
            rows = [r for r in rows if r.trace_id == filters["trace_id"]]
        if filters.get("kind_prefix"):
            rows = [r for r in rows if r.kind.startswith(filters["kind_prefix"])]
        return list(reversed(rows))[: filters.get("limit", 200)]


async def test_a_job_page_shows_what_happened_to_that_job_for_this_account():
    s = TrailStub()
    s.rows = [
        _row(1, "ui.click", "Clicked “Apply” on /controller"),
        _row(2, "job.status", "R3 marked “DevOps Engineer at Northwind Labs” applied for you; nothing was sent "
                              "to the employer", actor="system"),
        _row(3, "job.status", "Someone else's change", user_id=7),
    ]
    html = (await _call(s, "GET", f"/jobs/{JOB.id}", headers={"Accept": "text/html"})).text
    assert "What happened" in html
    assert "nothing was sent to the employer" in html
    assert "Clicked “Apply”" not in html  # clicks are in the admin trail, not the job's story
    assert "Someone else" not in html
    assert s.queries[-1]["user_id"] == 42 and s.queries[-1]["subject"] == f"job:{JOB.id}"


class LiveTrailStub(TrailStub):
    """TrailStub whose written events are readable back, as with Postgres."""

    async def insert_events(self, events):
        from r3.db import EventRow

        for e in events:
            self.rows.append(EventRow(id=len(self.rows) + 1, at=e.at, trace_id=e.trace_id, user_id=e.user_id,
                                      actor=e.actor, kind=e.kind, subject=e.subject, summary=e.summary, data=e.data))


async def test_a_status_change_is_on_the_job_page_right_after_the_redirect():
    # Events reach the database in one-second batches; the page reads its
    # history only after writing what is buffered, so "I applied" shows at once.
    s = LiveTrailStub()
    app = create_app(database=s, auth=True)
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.cookies.set(SESSION_COOKIE, "seeker")
        r = await c.post(f"/jobs/{JOB.id}/status", data={"status": "applied", "redirect": "1"},
                         follow_redirects=True)
    assert "You marked “Senior ML Engineer at Acme AI” applied" in r.text


async def test_the_events_page_is_for_admins_and_follows_one_trace():
    s = TrailStub()
    other = "1" * 32
    s.rows = [_row(1, "ui.click", "Clicked “Apply” on /controller"),
              _row(2, "chat.reply", "R3 answered: apply?", actor="ai", data={"model": "openrouter/free"}),
              _row(3, "page.view", "Opened /jobs", trace=other)]
    assert (await _call(s, "GET", "/events", headers={"Accept": "text/html"})).status_code == 403
    html = (await _call(s, "GET", f"/events?trace={TRACE}", token="admin", headers={"Accept": "text/html"})).text
    assert "Clicked “Apply” on /controller" in html and "R3 answered: apply?" in html
    assert "Opened /jobs" not in html
    assert "openrouter/free" in html  # the details are there to expand
    assert s.queries[-1]["trace_id"] == TRACE


async def test_a_form_post_takes_its_trace_from_the_click_cookie(stub):
    """A plain form post navigates, so the page cannot add a header; track.js
    leaves the click's trace in a short-lived cookie instead."""
    r = await _call(stub, "POST", f"/jobs/{JOB.id}/status", data={"status": "applied", "redirect": "1"},
                    cookies={"r3_trace": TRACE})
    assert r.headers["x-r3-trace"] == TRACE
    assert next(e for e in stub.events if e.kind == "job.status").trace_id == TRACE


async def test_the_header_wins_over_the_cookie(stub):
    r = await _call(stub, "POST", f"/jobs/{JOB.id}/status", data={"status": "applied"},
                    headers={"X-R3-Trace": TRACE}, cookies={"r3_trace": "0" * 32})
    assert r.headers["x-r3-trace"] == TRACE
