"""The resume job match: the SQL list_jobs generates with and without the
user's skills (asserted on strings and arguments with a recording pool), and
the pages a regular account and an admin see, against an in-memory stub."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from r3 import db
from r3.db.jobs import MATCH_SORT_SQL, JobQueries, match_order, match_sql, title_fit_sql
from r3.resume.roles import title_patterns
from r3.web import create_app
from r3.web.auth import User, current_user
from tests.test_routes import JOB, StubDB
from tests.test_user_jobs import RecordingPool

SKILLS = ["aws", "kubernetes", "python"]

# --- SQL builder -----------------------------------------------------------


def _queries(pool: RecordingPool) -> JobQueries:
    q = JobQueries()
    q.pool = pool  # type: ignore[assignment]
    return q


async def test_without_skills_the_sql_is_unchanged_and_match_falls_back():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(status="new", sort="newest"), 2, 25)
    count_sql, count_args = pool.calls[0]
    sql, args = pool.calls[1]
    assert "match_count" not in sql and "LATERAL" not in sql and "uj." not in sql
    assert "ORDER BY j.scraped_at DESC" in sql and "LIMIT $2 OFFSET $3" in sql
    assert args == ("new", 25, 25) and count_args == ("new",)
    # "match" without skills is an unknown sort: the default order, same SQL.
    await q.list_jobs(db.JobFilter(status="new", sort="match"), 2, 25)
    assert pool.calls[3] == (sql, args)


async def test_match_sort_binds_skills_after_the_filters_and_ranks_by_overlap():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(status="new", tech_tag="aws", sort="match"), 1, 50,
                      user_id=9001, skills=SKILLS)
    count_sql, count_args = pool.calls[0]
    sql, args = pool.calls[1]
    # The count is exactly the unmatched one: no skills, no lateral.
    assert "LATERAL" not in count_sql and count_args == (9001, "new", "aws")
    cols, join = match_sql(4)
    assert cols in sql and join in sql
    assert "t = ANY($4::text[])" in sql and "count(DISTINCT t)::int AS match_count" in sql
    assert "cardinality(COALESCE(j.tech_tags, '{}')) AS match_total" in sql
    assert f"ORDER BY {match_order()}" in sql and "LIMIT $5 OFFSET $6" in sql
    assert args == (9001, "new", "aws", SKILLS, 50, 0)
    # The skills are a bound array, never spliced into the SQL text.
    assert "kubernetes" not in sql


async def test_skills_with_another_sort_still_carry_the_counts():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(sort="salary"), 1, 50, user_id=9001, skills=SKILLS)
    sql, args = pool.calls[1]
    assert "m.match_count" in sql and "ORDER BY j.salary_max DESC NULLS LAST" in sql
    assert args == (9001, SKILLS, 50, 0)


def test_match_order_is_overlap_then_coverage_then_newest():
    assert MATCH_SORT_SQL.split(",")[0] == "m.match_count DESC"
    assert "/ NULLIF(cardinality(j.tech_tags), 0) DESC NULLS LAST" in MATCH_SORT_SQL
    assert MATCH_SORT_SQL.endswith("j.scraped_at DESC")


def test_match_order_puts_expired_postings_last_and_title_fit_before_overlap():
    # The shared status, not the user's triage: a posting that is gone never
    # outranks a live one, whatever its tags.
    live = "(COALESCE(j.last_validation, '') <> 'expired') DESC"
    assert match_order() == f"{live}, {MATCH_SORT_SQL}"
    fit = title_fit_sql(4, 5)
    assert fit == ("CASE WHEN j.position ~* ANY($4::text[]) THEN 2"
                   " WHEN j.position ~* ANY($5::text[]) THEN 1 ELSE 0 END")
    assert match_order(fit) == f"{live}, {fit} DESC, {MATCH_SORT_SQL}"


TITLES = (title_patterns(["devops", "platform"]), title_patterns(["linux"]))


async def test_match_sort_binds_title_patterns_after_the_skills():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(status="new", sort="match"), 1, 50, user_id=9001, skills=SKILLS,
                      titles=TITLES)
    count_sql, count_args = pool.calls[0]
    sql, args = pool.calls[1]
    assert "~*" not in count_sql and count_args == (9001, "new")
    assert f"ORDER BY {match_order(title_fit_sql(4, 5))}" in sql and "LIMIT $6 OFFSET $7" in sql
    assert args == (9001, "new", SKILLS, *TITLES, 50, 0)
    # The resume's roles are bound arrays, never spliced into the SQL text.
    assert "devops" not in sql and "linux" not in sql


async def test_title_patterns_bind_only_for_the_match_sort():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(sort="salary"), 1, 50, user_id=9001, skills=SKILLS, titles=TITLES)
    sql, args = pool.calls[1]
    assert "~*" not in sql and args == (9001, SKILLS, 50, 0)
    # Without skills there is no match sort at all, so no title fit either.
    await q.list_jobs(db.JobFilter(sort="match"), 1, 50, user_id=9001, titles=TITLES)
    sql, args = pool.calls[3]
    assert "~*" not in sql and "ORDER BY j.scraped_at DESC" in sql and args == (9001, 50, 0)


async def test_match_sort_without_title_keywords_keeps_the_overlap_order():
    pool = RecordingPool()
    q = _queries(pool)
    for titles in (None, ([], [])):
        await q.list_jobs(db.JobFilter(sort="match"), 1, 50, user_id=9001, skills=SKILLS, titles=titles)
        sql, args = pool.calls[-1]
        assert f"ORDER BY {match_order()}" in sql and "~*" not in sql and args == (9001, SKILLS, 50, 0)


async def test_vocabulary_query_reads_distinct_tags():
    pool = RecordingPool()
    assert await _queries(pool).tech_vocabulary() == []
    assert "SELECT DISTINCT tag FROM scraped_jobs, unnest(tech_tags) AS tag" in pool.calls[0][0]


# --- pages -----------------------------------------------------------------

RESUME = "Platform engineer: Python, Kubernetes and AWS. Terraform too."
SEEKER = User(id=1, account_name="alice")
ADMIN = User(id=0, account_name="local", is_admin=True)
XSS_JOB = JOB.model_copy(update={"id": 8, "tech_tags": ["python", "<script>alert(1)</script>"]})
NOW = datetime.now(timezone.utc)


class MatchDB(StubDB):
    def __init__(self, resume: str | None = RESUME) -> None:
        super().__init__()
        self.resume = resume
        self.vocab_loads = 0
        self.list_calls: list[dict] = []
        self.label_pool = [JOB, XSS_JOB]

    async def primary_resume(self, user_id):
        if self.resume is None:
            return None
        return db.Resume(id=3, user_id=user_id, title="Main", source="paste", is_primary=True,
                         created_at=NOW, updated_at=NOW, content=self.resume)

    async def list_resumes(self, user_id):
        return []

    async def tech_vocabulary(self):
        self.vocab_loads += 1
        return ["aws", "python", "kubernetes", "terraform", "go", "<script>alert(1)</script>"]

    async def list_jobs(self, f, page, size, *, user_id=None, **kw):
        self.list_calls.append({"sort": f.sort, "status": f.status, "size": size, **kw})
        await super().list_jobs(f, page, size, user_id=user_id)
        skills = kw.get("skills")
        jobs = [j.model_copy(update={"match_count": len(set(j.tech_tags) & set(skills)),
                                     "match_total": len(j.tech_tags)}) if skills else j
                for j in (JOB, XSS_JOB)]
        return jobs, db.Page(number=1, size=size, total=2, pages=1, has_prev=False, has_next=False)


@pytest.fixture
async def site():
    stub = MatchDB()
    app = create_app(database=stub)

    def as_user(user: User) -> None:
        app.dependency_overrides[current_user] = lambda: user

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, stub, as_user


async def test_seeker_with_resume_defaults_to_best_match_with_badges(site):
    client, stub, as_user = site
    as_user(SEEKER)
    page = await client.get("/jobs")
    assert page.status_code == 200
    assert stub.list_calls[-1]["sort"] == "match"
    assert stub.list_calls[-1]["skills"] == ["aws", "kubernetes", "python", "terraform"]
    assert '<option value="match"    selected>best match</option>' in page.text
    assert "2 of 2 skills" in page.text and "1 of 2 skills" in page.text
    assert '<span class="tag matched">python</span>' in page.text
    assert "Add your resume" not in page.text
    # A sort the user picked wins, and the counts still come along.
    await client.get("/jobs?sort=newest")
    assert stub.list_calls[-1]["sort"] == "newest" and "skills" in stub.list_calls[-1]
    # The vocabulary is cached across requests.
    assert stub.vocab_loads == 1


async def test_best_match_ranks_by_the_roles_the_resume_names(site):
    client, stub, as_user = site
    stub.resume = ("Maya Okafor\nDevOps / Platform Engineer\nmaya@example.com\n\nEXPERIENCE\n"
                   "Linux system administrator — Mercy Valley Health\nMarch 2019 – Present\n- Patching\n\n"
                   "SKILLS\nPython, AWS\n")
    as_user(SEEKER)
    await client.get("/jobs")
    assert stub.list_calls[-1]["sort"] == "match"
    assert stub.list_calls[-1]["titles"] == (title_patterns(["devops", "platform"]),
                                             title_patterns(["linux", "system"]))
    await client.get("/home")
    assert stub.list_calls[-1]["titles"] == (title_patterns(["devops", "platform"]),
                                             title_patterns(["linux", "system"]))
    # A resume that names no role ranks exactly as before.
    stub.resume = "Python and AWS."
    await client.get("/jobs")
    assert stub.list_calls[-1]["titles"] is None


async def test_tags_are_escaped(site):
    client, _, as_user = site
    as_user(SEEKER)
    text = (await client.get("/jobs")).text + (await client.get("/jobs/8")).text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text


async def test_seeker_without_resume_gets_the_hint_and_no_match(site):
    client, stub, as_user = site
    stub.resume = None
    as_user(SEEKER)
    page = await client.get("/jobs")
    assert 'href="/resume">Add your resume</a> to see which jobs match your skills.' in page.text
    assert "skills" not in stub.list_calls[-1] and stub.list_calls[-1]["sort"] == ""
    assert "match-badge" not in page.text and "best match" not in page.text


async def test_job_page_shows_matched_and_missing_tags(site):
    client, stub, as_user = site
    stub.resume = "Python only."
    as_user(SEEKER)
    page = await client.get("/jobs/7")
    assert "Your resume shows 1 of the 2 skills this posting lists." in page.text
    assert '<span class="tag matched">python</span>' in page.text
    assert '<span class="tag missing"><span class="sr-only">missing: </span>kubernetes</span>' in page.text


async def test_home_lists_best_matches_only_with_a_resume(site):
    client, stub, as_user = site
    as_user(SEEKER)
    page = await client.get("/home")
    assert "Best matches for you" in page.text and 'href="/jobs/7"' in page.text
    call = stub.list_calls[-1]
    assert (call["sort"], call["status"], call["size"]) == ("match", "new", 5)
    stub.resume = None
    calls = len(stub.list_calls)
    page = await client.get("/home")
    assert "Best matches for you" not in page.text and "Find jobs that fit" in page.text
    assert len(stub.list_calls) == calls


async def test_admin_pages_are_unchanged(site):
    client, stub, as_user = site
    as_user(ADMIN)
    jobs = await client.get("/jobs")
    detail = await client.get("/jobs/7")
    assert "skills" not in stub.list_calls[-1] and stub.list_calls[-1]["sort"] == ""
    assert "titles" not in stub.list_calls[-1]
    for text in (jobs.text, detail.text):
        assert "match-badge" not in text and "best match" not in text
        assert "Your resume shows" not in text and "to see which jobs match your skills" not in text
    assert stub.vocab_loads == 0
