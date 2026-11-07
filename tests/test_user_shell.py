"""The job seeker's shell: what a regular account sees on Home, in the nav,
on Explore, in the jobs table and on a job page, versus an admin (and the
sign-in-off local admin), who keep the operator console. Runs against an
in-memory stub with real sign-in (a session cookie), so the layout's
``viewer()`` and the handlers' ``current_user`` agree."""

from __future__ import annotations

import re

import httpx
import pytest

from r3 import db
from r3.db.jobs import SHARED_STATUS_SQL, USER_JOIN_SQL, USER_STATUS_SQL
from r3.db.resumes import Resume, ResumeSummary
from r3.db.scout import ScoutQueries
from r3.web import create_app
from r3.web.auth import SESSION_COOKIE, User
from tests.test_auth import AuthStubDB
from tests.test_routes import JOB, StubDB

PREFIX = "/dashboard/controller/r3"
SEEKER = User(id=42, account_name="Dana", is_admin=False)
ADMIN = User(id=1, account_name="boss", is_admin=True)
OPERATOR_HREFS = ("/label", "/settings")


def _resume(rid: int, title: str, source: str = "upload", primary: bool = False, job_id: int | None = None) -> Resume:
    return Resume(id=rid, user_id=SEEKER.id, title=title, source=source, is_primary=primary,
                  job_id=job_id, content="text")


class ShellStub(AuthStubDB):
    """AuthStubDB with fixed accounts behind two tokens, per-user numbers,
    resumes and job statuses, and a record of which user each count was for."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens = {"seeker": SEEKER, "admin": ADMIN}
        self.resumes: list[Resume] = []
        self.statuses: dict[int, str] = {}
        self.overview_users: list[int | None] = []
        self.scout_users: list[int | None] = []

    async def session_user(self, token):
        return self.tokens.get(token)

    async def overview(self, *, user_id=None):
        self.overview_users.append(user_id)
        stats = await super().overview()
        return stats.model_copy(update={"new_jobs": 37, "applied_jobs": 2}) if user_id else stats

    async def primary_resume(self, user_id):
        return next((r for r in self.resumes if r.is_primary), None)

    async def list_resumes(self, user_id):
        return [ResumeSummary(**r.model_dump(exclude={"content", "fit"})) for r in self.resumes]

    async def scout_presets(self, scope, *, user_id=None):
        self.scout_users.append(user_id)
        return await super().scout_presets(scope)

    async def scout_combos(self, scope, *, user_id=None):
        self.scout_users.append(user_id)
        return await super().scout_combos(scope)

    async def get_job(self, job_id, *, user_id=None):
        if job_id != JOB.id:
            return None
        return JOB.model_copy(update={"status": self.statuses.get(job_id, "new")})

    async def set_user_job_status(self, user_id, job_id, status):
        self.statuses[job_id] = status


@pytest.fixture
def stub():
    return ShellStub()


async def _get(app, path, token=None, base_url="http://test"):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url) as c:
            if token:
                c.cookies.set(SESSION_COOKIE, token)
            return await c.get(path)


async def _as(stub, token, path):
    r = await _get(create_app(database=stub, auth=True), path, token)
    assert r.status_code == 200, r.text[:300]
    return r.text


def _hrefs(html: str) -> set[str]:
    return set(re.findall(r'href="([^"]*)"', html))


# --- Home ----------------------------------------------------------------------


async def test_new_account_home_is_a_checklist_with_own_numbers(stub):
    html = await _as(stub, "seeker", "/home")
    assert "Hi, Dana" in html and "Get started" in html
    assert stub.overview_users == [42]
    # Own numbers, and the one marketing line about the shared corpus.
    assert ">37<" in html and "new for you" in html and ">2<" in html and "applied by you" in html
    assert "100 jobs from 4 job sites, newest added 1h ago." in html
    # Step 1 is open and points at both ways in.
    assert 'href="/resume"' in html and 'href="/resume/build"' in html
    assert "Build one by answering a few questions" in html
    assert 'aria-label="Done"' in html  # only step 3: the stub user has 2 applied
    assert html.count('aria-label="Done"') == 1
    assert 'href="/controller"' in html and 'href="/scout"' in html
    # Nothing from the operator overview.
    for leak in ("job sites</div>", "Example Boards", "By status", "skipped", "added last", "Top sites"):
        assert leak not in html, leak


async def test_home_checklist_follows_resume_state(stub):
    stub.resumes = [_resume(3, "Dana — Platform Engineer", primary=True)]
    html = await _as(stub, "seeker", "/home")
    assert 'href="/resume/3"' in html and "Dana — Platform Engineer" in html
    assert html.count('aria-label="Done"') == 2  # resume + applied


async def test_admin_and_sign_in_off_keep_the_operator_overview(stub):
    html = await _as(stub, "admin", "/")
    assert "job sites</div>" in html and "Get started" not in html
    assert stub.overview_users == [None]
    off = await _get(create_app(database=StubDB()), "/")
    assert off.status_code == 200 and "job sites</div>" in off.text and "Get started" not in off.text


async def test_account_name_and_resume_title_are_escaped(stub):
    stub.tokens["seeker"] = SEEKER.model_copy(update={"account_name": "<script>alert(1)</script>"})
    stub.resumes = [_resume(3, "<img src=x onerror=alert(2)>", primary=True)]
    html = await _as(stub, "seeker", "/home")
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<img src=x" not in html and "&lt;img src=x onerror=alert(2)&gt;" in html


async def test_home_links_carry_the_root_path(stub):
    app = create_app(database=stub, root_path=PREFIX, auth=True)
    r = await _get(app, "home", "seeker", base_url=f"http://test{PREFIX}/")
    assert r.status_code == 200
    hrefs = _hrefs(r.text)
    for path in ("/resume", "/resume/build", "/jobs?status=new", "/scout", "/controller", "/jobs?status=applied"):
        assert PREFIX + path in hrefs, path
    assert not [h for h in hrefs if h.startswith("/") and not h.startswith(PREFIX)]


# --- Nav -----------------------------------------------------------------------


async def test_nav_by_role(stub):
    seeker = await _as(stub, "seeker", "/jobs")
    nav = seeker.split('<nav aria-label="Main">')[1].split("</nav>")[0]
    assert [a for a in re.findall(r">([A-Za-z]+)</a>", nav)] == ["Assistant", "Jobs", "Resume"]
    assert not any(h in _hrefs(seeker) for h in OPERATOR_HREFS)
    assert "job search" in seeker
    assert "Sign out" in seeker and "Dana" in seeker

    admin = await _as(stub, "admin", "/jobs")
    ops = admin.split('class="nav-ops"')[1].split("</span>")[0]
    assert all(f'href="{h}"' in ops for h in OPERATOR_HREFS)
    off = await _get(create_app(database=StubDB()), "/jobs")
    assert 'class="nav-ops"' in off.text and "Assistant" in off.text


# --- Explore -------------------------------------------------------------------


async def test_explore_counts_are_the_users_own(stub):
    html = await _as(stub, "seeker", "/scout")
    assert stub.scout_users == [42, 42]
    assert "Explore jobs" in html and "Ready-made searches over the stored jobs" in html
    assert "Popular searches" in html

    stub.scout_users.clear()
    admin = await _as(stub, "admin", "/scout")
    assert stub.scout_users == [1, 1]
    assert "Explore jobs" in admin and "Curated hunts" in admin


class _Pool:
    def __init__(self) -> None:
        self.sql: list[tuple[str, tuple]] = []

    async def fetchval(self, sql, *args):
        self.sql.append((sql, args))
        return 0

    async def fetch(self, sql, *args):
        self.sql.append((sql, args))
        return []

    async def fetchrow(self, sql, *args):
        self.sql.append((sql, args))
        return {k: 0 for k in re.findall(r"AS (\w+)", sql)}


def _queries(pool: _Pool) -> ScoutQueries:
    q = ScoutQueries()
    q.pool = pool  # type: ignore[assignment]
    return q


async def test_scout_sql_binds_the_user_and_keeps_the_shared_form():
    shared, mine = _Pool(), _Pool()
    await _queries(shared).scout_presets(db.ScoutScope())
    await _queries(shared).scout_combos(db.ScoutScope())
    await _queries(mine).scout_presets(db.ScoutScope(), user_id=42)
    await _queries(mine).scout_combos(db.ScoutScope(), user_id=42)
    assert shared.sql[0] == (f"SELECT count(*) FROM scraped_jobs j WHERE {SHARED_STATUS_SQL} = $1", ("new",))
    assert all("uj." not in s for s, _ in shared.sql)
    assert mine.sql[0] == (f"SELECT count(*) FROM scraped_jobs j {USER_JOIN_SQL} WHERE {USER_STATUS_SQL} = $2", (42, "new"))
    assert all(args[0] == 42 and USER_JOIN_SQL in s for s, args in mine.sql)


# --- Jobs table and job page ---------------------------------------------------


async def test_seeker_row_actions_view_applied_not_for_me_and_undo(stub):
    html = await _as(stub, "seeker", "/jobs")
    row = html.split('id="job-7"')[1].split("</tr>")[0]
    assert 'href="/jobs/7">View<span class="sr-only">' in row and ">Applied<span" in row and ">Not for me<span" in row
    assert "expire" not in row and "reset" not in row and "Undo" not in row
    # No per-site dropdown or site links for a job seeker.
    assert 'name="site"' not in html and "/jobs?site=" not in row

    app = create_app(database=stub, auth=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            c.cookies.set(SESSION_COOKIE, "seeker")
            r = await c.post("/jobs/7/status", data={"status": "skipped"}, headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert ">Undo<span" in r.text and '"status":"new"' in r.text
    assert ">Not for me<span" not in r.text and "not for me" in r.text

    admin = await _as(stub, "admin", "/jobs")
    assert ">expire</button>" in admin and ">View<" not in admin and '<select name="site"' in admin


async def test_job_page_hides_operator_metadata_from_seekers(stub):
    html = await _as(stub, "seeker", "/jobs/7")
    # The posting itself is shown while R3_JOB_POSTINGS=full (tests/test_job_postings.py); the operator's
    # metadata never is.
    for leak in ("Validation", "Verification", "Job ID", "Scraped", "site: ", "/jobs?site=", '<select name="status">'):
        assert leak not in html, leak
    for keep in ("Senior ML Engineer", "Acme AI", "Open posting", "Employment", "Salary", "Posted", "Save notes",
                 ">I applied</button>", ">Not for me</button>"):
        assert keep in html, keep

    admin = await _as(stub, "admin", "/jobs/7")
    assert "Validation" in admin and "Job ID" in admin and '<select name="status">' in admin


async def test_forbidden_and_error_pages_show_the_viewers_nav_and_a_home_link(stub):
    app = create_app(database=stub, auth=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            c.cookies.set(SESSION_COOKIE, "seeker")
            pages = [await c.get(p, headers={"Accept": "text/html"}) for p in ("/settings", "/no-such-page")]
    assert [r.status_code for r in pages] == [403, 404]
    for r in pages:
        nav = r.text.split('<nav aria-label="Main">')[1].split("</nav>")[0]
        assert re.findall(r">([A-Za-z]+)</a>", nav) == ["Assistant", "Jobs", "Resume"]
        assert not any(h in _hrefs(r.text) for h in OPERATOR_HREFS)
        assert re.search(r'href="/"[^>]*>Back to home<', r.text)


def test_wrapped_phone_topbar_does_not_offset_sticky_headers():
    from r3.web.templating import STATIC_DIR

    css = (STATIC_DIR / "app.css").read_text()
    phone = css[css.rindex("@media (max-width: 700px)"):]
    phone = phone[:phone.index("\n}\n")]
    assert ".topbar { position: static; }" in phone and ".grid th, .panel-tabs { top: 0; }" in phone


def test_home_checklist_grid_track_can_shrink():
    """The checklist is a one-column grid; its track must be allowed to shrink
    below the content's max-content width. A long best-match title once pushed
    the auto track to 395px and the page overflowed its 390px viewport."""
    import re

    from r3.web.templating import STATIC_DIR

    rule = re.search(r"\.checklist \{[^}]*\}", (STATIC_DIR / "app.css").read_text()).group(0)
    assert "grid-template-columns: minmax(0, 1fr)" in rule
