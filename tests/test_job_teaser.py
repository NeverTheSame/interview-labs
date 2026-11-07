"""The job teaser (OQ-6): a regular account sees that a role exists, that it
is live, its stack, pay and location, and on request a short overview, but
never the posting itself, its links or the board it came from. Admins keep
the full posting. Runs against the shell stub with real sign-in."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta

import httpx
import pytest

from r3 import db
from r3.db.job_summaries import JobSummaryQueries
from r3.db.jobs import job_columns
from r3.web import create_app
from r3.web.auth import SESSION_COOKIE
from tests.fakes import FakeSummarizer
from tests.test_routes import JOB, NOW
from tests.test_user_shell import PREFIX, ShellStub

TEASER_JOB = JOB.model_copy(update={"apply_url": "https://apply.acme.example/7", "last_validation": "alive",
                                    "last_validated_at": NOW - timedelta(days=2)})
# What a seeker's job page must never carry.
LEAKS = (JOB.job_url, "boards.example", "apply.acme.example", "Example Boards", "Open posting", "Build models",
         "Ship them", ">Links<", ">Description<")


class TeaserStub(ShellStub):
    def __init__(self) -> None:
        super().__init__()
        self.job = TEASER_JOB
        self.ensured = False

    async def get_job(self, job_id, *, user_id=None):
        if job_id != self.job.id:
            return None
        return self.job.model_copy(update={"status": self.statuses.get(job_id, self.job.status)})

    async def ensure_job_summary_table(self):
        self.ensured = True


@pytest.fixture(autouse=True)
def _teaser_mode(monkeypatch):
    """These tests are about the teaser, which R3_JOB_POSTINGS=teaser turns on."""
    monkeypatch.setenv("R3_JOB_POSTINGS", "teaser")


HX = {"HX-Request": "true"}


@asynccontextmanager
async def _client(stub, summarizer=None, token="seeker"):
    """One app for several requests, so its in-memory state carries over."""
    app = create_app(database=stub, auth=True,
                     job_summarizer=summarizer if summarizer is not None else FakeSummarizer())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test/") as c:
            c.cookies.set(SESSION_COOKIE, token)
            yield c


async def _call(stub, method, path, token, *, summarizer=None, root_path="", htmx=False):
    app = create_app(database=stub, auth=True, root_path=root_path,
                     job_summarizer=summarizer if summarizer is not None else FakeSummarizer())
    headers = {"HX-Request": "true"} if htmx else {}
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url=f"http://test{root_path}/") as c:
            c.cookies.set(SESSION_COOKIE, token)
            return await c.request(method, path.lstrip("/"), headers=headers)


# --- the job page ----------------------------------------------------------


async def test_seeker_job_page_is_a_teaser_without_the_posting_or_its_links():
    stub, summarizer = TeaserStub(), FakeSummarizer()
    r = await _call(stub, "GET", "/jobs/7", "seeker", summarizer=summarizer)
    assert r.status_code == 200
    for leak in LEAKS:
        assert leak not in r.text, leak
    for keep in ("Senior ML Engineer", "Acme AI", "About this role", 'action="/jobs/7/overview"',
                 'hx-post="/jobs/7/overview"', ">python<", ">kubernetes<", "Remote, US", "Salary", "Save notes",
                 ">I applied</button>"):
        assert keep in r.text, keep
    assert summarizer.calls == 0  # opening the page never calls the model
    assert stub.ensured


async def test_admin_job_page_keeps_the_full_posting():
    r = await _call(TeaserStub(), "GET", "/jobs/7", "admin")
    for keep in ("Open posting", JOB.job_url, "apply.acme.example", "Build models.", ">Description<", ">Links<"):
        assert keep in r.text, keep
    assert "About this role" not in r.text


async def test_live_shows_when_the_job_was_last_checked_and_an_expired_one_is_not_live():
    stub = TeaserStub()
    r = await _call(stub, "GET", "/jobs/7", "seeker")
    assert "Live" in r.text and "checked 2d ago" in r.text and "No longer live" not in r.text

    stub.job = TEASER_JOB.model_copy(update={"last_validation": "", "last_validated_at": None})
    r = await _call(stub, "GET", "/jobs/7", "seeker")
    assert "Live" in r.text and "checked" not in r.text

    stub.job = TEASER_JOB.model_copy(update={"live": False})
    r = await _call(stub, "GET", "/jobs/7", "seeker")
    assert "No longer live" in r.text


def test_live_reads_the_posting_in_both_views():
    for cols in (job_columns(None), job_columns(42)):
        assert "(COALESCE(j.last_validation, '') <> 'expired') AS live" in cols


def test_database_carries_the_summary_queries():
    assert issubclass(db.Database, JobSummaryQueries)


# --- POST /jobs/{id}/overview ----------------------------------------------


async def test_overview_is_generated_once_then_served_from_the_cache():
    stub, summarizer = TeaserStub(), FakeSummarizer()
    r = await _call(stub, "POST", "/jobs/7/overview", "seeker", summarizer=summarizer, htmx=True)
    assert r.status_code == 200 and "<html" not in r.text
    assert summarizer.text in r.text
    assert (stub.summaries[7].summary, stub.summaries[7].model) == (summarizer.text, "fake-model")

    r = await _call(stub, "POST", "/jobs/7/overview", "seeker", summarizer=summarizer, htmx=True)
    assert summarizer.text in r.text and summarizer.calls == 1

    page = await _call(stub, "GET", "/jobs/7", "seeker", summarizer=summarizer)
    assert summarizer.text in page.text and 'action="/jobs/7/overview"' not in page.text


async def test_a_fallback_is_shown_never_cached_and_not_retried_for_a_while(monkeypatch):
    from r3.web import routes_overview

    stub, summarizer = TeaserStub(), FakeSummarizer(text="Build models.", engine="extract")
    async with _client(stub, summarizer) as c:
        r = await c.post("jobs/7/overview", headers=HX)
        assert "Build models." in r.text and stub.summaries == {}
        r = await c.post("jobs/7/overview", headers=HX)
        assert "Build models." in r.text and summarizer.calls == 1  # no second model call in the window
        page = await c.get("jobs/7")
        assert "Build models." in page.text and 'action="/jobs/7/overview"' not in page.text
        r = await c.post("jobs/7/overview")  # without JS: back to the page, which shows the fallback
        assert r.status_code == 303 and r.headers["location"] == "/jobs/7#about"

        monkeypatch.setattr(routes_overview, "FALLBACK_TTL_S", 0.0)
        await c.post("jobs/7/overview", headers=HX)
        assert summarizer.calls == 2  # the window is over: the model is asked again


async def test_no_query_parameter_shows_the_posting_text_without_a_click():
    page = await _call(TeaserStub(), "GET", "/jobs/7?about=1", "seeker")
    assert "Build models" not in page.text and 'action="/jobs/7/overview"' in page.text


async def test_concurrent_clicks_on_one_job_share_one_model_call():
    import asyncio

    class Gate(FakeSummarizer):
        def __init__(self) -> None:
            super().__init__()
            self.open = asyncio.Event()

        async def overview(self, job):
            await self.open.wait()
            return await super().overview(job)

    summarizer = Gate()
    async with _client(TeaserStub(), summarizer) as c:
        first = asyncio.ensure_future(c.post("jobs/7/overview", headers=HX))
        second = asyncio.ensure_future(c.post("jobs/7/overview", headers=HX))
        await asyncio.sleep(0.05)
        summarizer.open.set()
        responses = await asyncio.gather(first, second)
    assert summarizer.calls == 1 and all(summarizer.text in r.text for r in responses)


async def test_overview_without_htmx_redirects_back_under_the_root_path():
    r = await _call(TeaserStub(), "POST", "/jobs/7/overview", "seeker", root_path=PREFIX)
    assert r.status_code == 303 and r.headers["location"] == f"{PREFIX}/jobs/7#about"
    page = await _call(TeaserStub(), "GET", "/jobs/7", "seeker", root_path=PREFIX)
    assert f'action="{PREFIX}/jobs/7/overview"' in page.text


async def test_overview_of_a_missing_job_is_404():
    r = await _call(TeaserStub(), "POST", "/jobs/999/overview", "seeker", htmx=True)
    assert r.status_code == 404


async def test_overview_text_is_escaped():
    summarizer = FakeSummarizer(text="<script>alert(1)</script> A role that ships models to production.")
    r = await _call(TeaserStub(), "POST", "/jobs/7/overview", "seeker", summarizer=summarizer, htmx=True)
    assert "<script>alert(1)" not in r.text and "&lt;script&gt;alert(1)&lt;/script&gt;" in r.text


async def test_a_broken_cache_never_breaks_the_page_or_the_overview():
    class Broken(TeaserStub):
        async def get_job_summary(self, job_id):
            raise ConnectionError("down")

        async def save_job_summary(self, job_id, summary, model):
            raise ConnectionError("down")

    summarizer = FakeSummarizer()
    page = await _call(Broken(), "GET", "/jobs/7", "seeker", summarizer=summarizer)
    assert page.status_code == 200 and 'action="/jobs/7/overview"' in page.text
    r = await _call(Broken(), "POST", "/jobs/7/overview", "seeker", summarizer=summarizer, htmx=True)
    assert r.status_code == 200 and summarizer.text in r.text


# --- everywhere else a job reaches a seeker ----------------------------------


async def test_seeker_jobs_table_has_no_source_column_or_site_sort():
    for htmx in (False, True):
        r = await _call(TeaserStub(), "GET", "/jobs", "seeker", htmx=htmx)
        assert ">Source<" not in r.text and "Example Boards" not in r.text and 'value="site"' not in r.text
    r = await _call(TeaserStub(), "POST", "/jobs/7/status?status=applied", "seeker", htmx=True)
    assert "Example Boards" not in r.text
    admin = await _call(TeaserStub(), "GET", "/jobs", "admin")
    assert ">Site<" in admin.text and "Example Boards" in admin.text and 'value="site"' in admin.text


async def test_api_job_cards_carry_no_link_or_board_for_a_seeker():
    card = (await _call(TeaserStub(), "GET", "/api/jobs", "seeker")).json()["jobs"][0]
    assert (card["job_url"], card["site_name"]) == ("", "")
    card = (await _call(TeaserStub(), "GET", "/api/jobs", "admin")).json()["jobs"][0]
    assert (card["job_url"], card["site_name"]) == (JOB.job_url, JOB.site_name)


async def test_chat_job_lists_carry_no_link_for_a_seeker():
    from r3.agent.actions import plan_action
    from r3.agent.contracts import JobsAction, RouteDecision
    from tests.fakes import FakeRuns
    from tests.fakes import test_settings as make_settings
    from tests.test_agent_users import ADMIN, DANA
    from tests.test_routes import StubDB

    async def plan(intent, user, **params):
        return await plan_action(RouteDecision(intent=intent, router="rules", params=params), database=StubDB(),
                                 runs=FakeRuns(), settings=make_settings(job_postings="teaser"), model_id="openrouter:openrouter/free",
                                 text=intent, user=user)

    for action in (await plan("jobs", DANA), await plan("jobs", DANA, job_ids=[JOB.id])):
        assert isinstance(action, JobsAction) and action.jobs
        assert all((c.job_url, c.site_name) == ("", "") for c in action.jobs)
    for user in (ADMIN, None):
        action = await plan("jobs", user)
        assert (action.jobs[0].job_url, action.jobs[0].site_name) == (JOB.job_url, JOB.site_name)


async def test_job_details_tool_gives_a_seeker_the_overview_not_the_posting():
    from tests.test_agent_users import ADMIN, DANA, RecordingDB, run_turn, tool_returns

    database = RecordingDB()
    _, seen = await run_turn(DANA, database, text="tell me about job 7", job_postings="teaser")
    out = tool_returns(seen)["job_details"]
    for leak in (JOB.job_url, "boards.example", "Example Boards", "URL:", "Site:", "Description:"):
        assert leak not in out, leak
    # The extract: half of this two-sentence posting, never all of it.
    assert "Overview: Build models." in out and "Ship them" not in out and "Senior ML Engineer" in out

    database.summaries[7] = db.JobSummary(job_id=7, summary="Acme AI ships models to production.", model="m")
    _, seen = await run_turn(DANA, database, text="tell me about job 7", job_postings="teaser")
    assert "Overview: Acme AI ships models to production." in tool_returns(seen)["job_details"]

    _, seen = await run_turn(ADMIN, RecordingDB(), text="tell me about job 7")
    out = tool_returns(seen)["job_details"]
    assert f"URL: {JOB.job_url}" in out and "Site: Example Boards" in out and "Description: Build models." in out
