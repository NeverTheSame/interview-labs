"""Browser end-to-end suite: the real app, a real Postgres, a real Chromium.

Set ``R3_E2E_PG_URL`` to an ADMIN Postgres URL (a throwaway cluster). The
session creates ``r3_e2e_<random>``, loads ``fixtures/schema.sql`` and
``fixtures/seed.sql``, serves ``create_app`` on a free localhost port from a
background thread with deterministic fakes for every model and network call,
and drops the database at the end. Without the variable (or with an
unreachable server) every test here is skipped.

Run: ``R3_E2E_PG_URL=postgresql://r3@127.0.0.1:54329/postgres uv run pytest tests/e2e``
"""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import socket
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PG_ENV = "R3_E2E_PG_URL"
OPERATOR = ("e2e-operator", "operator-pass")
OVERVIEW_TEXT = "Scripted overview: a writing role on a python team that ships small changes often."
PASSWORD = "e2e-pass-12345"  # new accounts need 12+ characters (routes_auth.PASSWORD_MIN)
SCRIPTED_REPLY = "Scripted reply: three DevOps roles look promising."

# Environment the app could read. All removed for the session so neither the
# live DATABASE_URL in .env nor a real model key can reach the app under test.
_SCRUB = re.compile(r"^(DATABASE_URL|R3_.*|TEMPORAL_.*|OPENROUTER.*|OPENAI.*|ANTHROPIC.*|TYPESAFE.*|"
                    r"LOGFIRE.*|GEMINI.*|GOOGLE_API.*|GROQ.*|MISTRAL.*)$")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    for item in items:
        if here in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.e2e)


def _with_db(url: str, name: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, "/" + name, p.query, p.fragment))


async def _admin(url: str, *sql: str) -> None:
    import asyncpg

    conn = await asyncpg.connect(url, timeout=3)
    try:
        for stmt in sql:
            await conn.execute(stmt)
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def e2e_dsn() -> Iterator[str]:
    admin = os.environ.get(PG_ENV, "").strip()
    if not admin:
        pytest.skip(f"{PG_ENV} is not set: browser e2e tests need an admin Postgres URL")
    try:
        asyncio.run(_admin(admin, "SELECT 1"))
    except Exception as e:  # unreachable, auth, DNS
        pytest.skip(f"{PG_ENV} is unreachable ({type(e).__name__}: {e})")
    name = f"r3_e2e_{secrets.token_hex(6)}"
    asyncio.run(_admin(admin, f'CREATE DATABASE "{name}"'))
    dsn = _with_db(admin, name)
    try:
        asyncio.run(_admin(dsn, (FIXTURES / "schema.sql").read_text(), (FIXTURES / "seed.sql").read_text()))
        yield dsn
    finally:
        asyncio.run(_admin(admin, f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


@pytest.fixture(scope="session")
def scrubbed_env(e2e_dsn: str) -> Iterator[None]:
    saved = dict(os.environ)
    for key in [k for k in os.environ if _SCRUB.match(k)]:
        del os.environ[key]
    os.environ.update({"DATABASE_URL": e2e_dsn, "R3_EMBED_WORKER": "0",
                       "R3_ADMIN_ACCOUNTS": "", "R3_AUTH_RATE_LIMIT": "100000"})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


_E2E_RUNS: "FakeRuns | None" = None
_E2E_APP = None


def e2e_runs() -> "FakeRuns":
    """The app's run service. A test may complete its runs (with a crafted
    result) to drive run cards the way the in-process runner would."""
    assert _E2E_RUNS is not None, "the app under test is not up"
    return _E2E_RUNS


async def _create_operator(dsn: str) -> None:
    from r3.db import Database

    db = await Database.open(dsn)
    try:
        await db.create_user(OPERATOR[0], OPERATOR[1], is_admin=True)
    finally:
        await db.close()


@pytest.fixture
def scripted_events():
    """Let one test choose what the scripted advisor streams for every turn;
    the default reply comes back afterwards."""
    assert _E2E_APP is not None, "the app under test is not up"
    advisor = _E2E_APP.state.advisor
    before = list(advisor.events)

    def use(events: list[object]) -> None:
        advisor.events = list(events)

    yield use
    advisor.events = before


@pytest.fixture
def teaser_mode():
    """Serve regular accounts the job teaser (R3_JOB_POSTINGS=teaser) for one
    test; the app's default, like the code's, is the full posting."""
    import dataclasses

    assert _E2E_APP is not None, "the app under test is not up"
    before = _E2E_APP.state.settings
    _E2E_APP.state.settings = dataclasses.replace(before, job_postings="teaser")
    yield
    _E2E_APP.state.settings = before


def _build_app(dsn: str):
    global _E2E_RUNS, _E2E_APP
    from r3.agent.events import DeltaEvent, DoneEvent
    from r3.resume.builder import ScriptedInterviewer
    from r3.resume.parse import HeuristicResumeParser
    from r3.settings import ControllerSettings
    from r3.web import create_app
    from tests.fakes import FakeRuns, FakeSummarizer, ScriptedAdvisor

    # A fake OpenAI key only so the chat composer is enabled; the advisor is
    # scripted, so nothing is ever sent. No OpenRouter key: that would let
    # /models?live=1 fetch the public catalogue from the internet.
    settings = ControllerSettings(api_keys={"openai": "sk-e2e-not-a-real-key"},
                                  default_model="openai:gpt-4o-mini", router_mode="rules")
    advisor = ScriptedAdvisor(settings, [DeltaEvent(text=SCRIPTED_REPLY), DoneEvent()])
    _E2E_RUNS = FakeRuns()
    # The overview summarizer answers like a model ("llm"), so the job
    # teaser's cache (r3_job_summaries) is exercised on real Postgres.
    _E2E_APP = create_app(dsn=dsn, settings=settings, runs=_E2E_RUNS, advisor=advisor, auth=True,
                          resume_interviewer=ScriptedInterviewer(),
                          job_summarizer=FakeSummarizer(text=OVERVIEW_TEXT),
                          # The fake OpenAI key would send the reader to the network; the heuristic reads offline.
                          resume_parser=HeuristicResumeParser())
    return _E2E_APP


@pytest.fixture(scope="session")
def base_url(e2e_dsn: str, scrubbed_env: None) -> Iterator[str]:
    """The app under test, served by uvicorn on a free port in a thread."""
    import httpx
    import uvicorn

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(_build_app(e2e_dsn), log_level="warning", lifespan="on"))
    errors: list[BaseException] = []

    def serve() -> None:
        try:
            server.run(sockets=[sock])
        except BaseException as e:  # surfaced below instead of a silent hang
            errors.append(e)

    thread = threading.Thread(target=serve, name="r3-e2e-server", daemon=True)
    thread.start()
    ready = threading.Event()
    for _ in range(300):  # 15 s budget, condition-polled
        if server.started or errors or not thread.is_alive():
            break
        ready.wait(0.05)
    if not server.started:
        server.should_exit = True
        raise RuntimeError(f"app under test did not start: {errors or 'timeout'}")
    # The operator is made out of band (public sign-up never grants admin).
    asyncio.run(_create_operator(os.environ["DATABASE_URL"]))
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        sock.close()


@dataclass
class Tracked:
    """Console errors and uncaught page errors seen by one page."""
    console_errors: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    dialogs: list[str] = field(default_factory=list)


@pytest.fixture
def make_page(browser, base_url: str) -> Iterator[Callable[..., object]]:
    """Open a fresh, isolated browser context and page. Requests to anything
    but the app are aborted, so the suite never touches the internet."""
    contexts = []
    host = urlsplit(base_url).netloc

    def factory(viewport: dict | None = None):
        ctx = browser.new_context(base_url=base_url, viewport=viewport or {"width": 1280, "height": 900},
                                  accept_downloads=True)
        ctx.set_default_timeout(10_000)
        ctx.route(re.compile(r"^(?!https?://%s/).*" % re.escape(host)),
                  lambda route: route.abort() if route.request.url.startswith("http") else route.continue_())
        contexts.append(ctx)
        page = ctx.new_page()
        tracked = Tracked()
        page.on("console", lambda m: tracked.console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: tracked.page_errors.append(str(e)))
        page.on("dialog", lambda d: (tracked.dialogs.append(d.message), d.dismiss()))
        page.tracked = tracked  # type: ignore[attr-defined]
        return page

    yield factory
    for ctx in contexts:
        ctx.close()


@pytest.fixture
def page(make_page):
    return make_page()


def unique_name(prefix: str = "u") -> str:
    return f"{prefix}-{secrets.token_hex(4)}"


def api_signup(page, name: str | None = None, password: str = PASSWORD) -> str:
    """Create an account and sign the page's context in, without the UI."""
    name = name or unique_name()
    r = page.request.post("/signup", form={"account_name": name, "password": password,
                                           "password_confirm": password, "next": ""})
    assert r.ok, f"signup failed: {r.status} {r.text()[:300]}"
    return name


def api_login(page, name: str, password: str) -> None:
    r = page.request.post("/login", form={"account_name": name, "password": password, "next": ""})
    assert r.ok, f"login failed: {r.status}"


def api_paste_resume(page, title: str, content: str) -> None:
    r = page.request.post("/resume/paste", form={"title": title, "content": content})
    assert r.ok, f"paste failed: {r.status}"


RESUME_TEXT = ("Jordan Example\nSite Reliability Engineer\n\nExperience\nBluefjord Systems, SRE, 2021-present\n"
               "- Cut paging volume by 40% with symptom-based Prometheus alerts on Kubernetes.\n"
               "- Wrote Terraform modules for AWS and Go tooling for deploys.\n\nSkills\nKubernetes, Terraform, "
               "AWS, Prometheus, Go, Python, Linux\n")
