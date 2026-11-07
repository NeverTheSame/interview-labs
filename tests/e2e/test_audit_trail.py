"""The audit trail in a real browser on real Postgres: a click, the request
it sends and the decision it makes land in r3_events under one trace id,
the job page tells the story, and nothing typed into a field is recorded."""

from __future__ import annotations

import asyncio
import re
import time
from concurrent.futures import ThreadPoolExecutor

from playwright.sync_api import expect

from tests.e2e.conftest import OPERATOR, api_login, api_signup
from tests.e2e.test_jobs_and_isolation import _open_job


async def _fetch(dsn: str, sql: str, *args):
    import asyncpg

    conn = await asyncpg.connect(dsn, timeout=3)
    try:
        return [dict(r) for r in await conn.fetch(sql, *args)]
    finally:
        await conn.close()


def _events(dsn: str, account: str, wait_for: str, timeout: float = 8.0, text: str = "") -> list[dict]:
    """The account's events, once one of kind ``wait_for`` (whose summary has
    ``text``) has been written (the portal writes in batches once a second)."""
    sql = ("SELECT e.kind, e.actor, e.trace_id, e.subject, e.summary, e.data::text AS data FROM r3_events e "
           "JOIN r3_users u ON u.id = e.user_id WHERE u.account_name = $1 ORDER BY e.id")
    deadline = time.monotonic() + timeout
    while True:
        # Playwright's sync API owns this thread's event loop; query from another.
        with ThreadPoolExecutor(1) as pool:
            rows = pool.submit(asyncio.run, _fetch(dsn, sql, account)).result()
        if any(r["kind"] == wait_for and text in r["summary"] for r in rows) or time.monotonic() > deadline:
            return rows
        time.sleep(0.3)


def test_a_click_its_request_and_its_decision_share_one_trace(page, e2e_dsn):
    name = api_signup(page)
    _open_job(page, "Observability Engineer")
    page.locator("textarea[name=notes]").fill("my secret plan: hunter22")  # typed, never recorded
    page.get_by_role("button", name="I applied").click()
    expect(page.locator(".pill.applied").first).to_have_text("applied")

    rows = _events(e2e_dsn, name, "ui.click", text="I applied")
    click = next(r for r in rows if r["kind"] == "ui.click" and "I applied" in r["summary"])
    post = next(r for r in rows if r["kind"] == "http.post" and "/status" in r["summary"])
    change = next(r for r in rows if r["kind"] == "job.status")
    assert click["trace_id"] == post["trace_id"] == change["trace_id"]
    assert change["actor"] == "user" and "Observability Engineer" in change["summary"]
    assert any(r["kind"] == "page.view" for r in rows)
    assert not any("hunter22" in r["data"] or "hunter22" in r["summary"] for r in rows)

    page.reload()
    history = page.locator("ol.history")
    expect(history).to_contain_text("You marked “Observability Engineer")
    expect(history).not_to_contain_text("Clicked")
    assert page.tracked.console_errors == [] and page.tracked.page_errors == []


def test_the_operator_follows_a_trace_on_the_events_page(page, e2e_dsn):
    api_login(page, *OPERATOR)
    _events(e2e_dsn, OPERATOR[0], "auth.signin")  # written within a second
    page.goto("/events?kind=auth.signin")
    expect(page.get_by_role("heading", name="Events")).to_be_visible()
    expect(page.locator("table.events")).to_contain_text(f"{OPERATOR[0]} signed in")
    page.locator("table.events td:last-child a").first.click()  # the trace link
    expect(page).to_have_url(re.compile(r"[?&]trace=[0-9a-f]{32}"))
    expect(page.locator("table.events tbody tr").first).to_be_visible()
    assert page.tracked.console_errors == [] and page.tracked.page_errors == []
