"""Per-user job triage (r3_user_jobs). The SQL half asserts on the generated
SQL and arguments, like test_jobs does for build_job_where, with a recording
pool instead of a database. The route half runs the web pages against an
in-memory overlay and switches accounts through ``current_user``."""

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from r3 import db
from r3.db.jobs import (_JOB_COLUMNS, SHARED_STATUS_SQL, USER_JOIN_SQL, USER_STATUS_SQL, JobQueries, build_job_where,
                        job_columns)
from r3.web import create_app
from r3.web.auth import User, current_user
from tests.test_routes import JOB, StubDB

# --- SQL builder -----------------------------------------------------------


def test_shared_where_filters_the_shared_status_without_a_user():
    clause, args = build_job_where(db.JobFilter(status="skipped", search="kafka"))
    assert clause == ("WHERE (j.position ILIKE $1 ESCAPE '\\' OR j.company ILIKE $1 ESCAPE '\\'"
                      f" OR j.description ILIKE $1 ESCAPE '\\') AND {SHARED_STATUS_SQL} = $2")
    assert args == ["%kafka%", "skipped"]
    assert "uj." not in clause


def test_user_where_binds_user_first_and_filters_effective_status():
    clause, args = build_job_where(db.JobFilter(status="skipped", search="kafka", days=7), user_id=9001)
    # $1 is the overlay join's user; filter values follow from $2.
    assert args == [9001, "%kafka%", "skipped", 7]
    assert "j.position ILIKE $2" in clause
    assert f"{USER_STATUS_SQL} = $3" in clause
    assert "make_interval(days => $4)" in clause
    assert "j.status = $" not in clause


def test_user_where_without_filters_still_binds_the_user():
    clause, args = build_job_where(db.JobFilter(), user_id=5)
    assert clause == "" and args == [5]


def test_effective_status_only_lets_an_expired_posting_through():
    assert SHARED_STATUS_SQL == "CASE WHEN (COALESCE(j.last_validation, '') <> 'expired') THEN 'new' ELSE 'expired' END"
    assert USER_STATUS_SQL == f"COALESCE(uj.status, {SHARED_STATUS_SQL})"
    assert USER_JOIN_SQL == "LEFT JOIN r3_user_jobs uj ON uj.job_id = j.id AND uj.user_id = $1"


def test_user_columns_read_triage_fields_from_the_overlay():
    assert job_columns(None) is _JOB_COLUMNS
    cols = job_columns(9001)
    assert f"{USER_STATUS_SQL} AS status" in cols
    assert "COALESCE(uj.notes, '') AS notes" in cols
    assert "uj.applied_at AS applied_at" in cols
    assert "'' AS notes" not in cols and "NULL::timestamptz AS applied_at" not in cols
    # Everything else is the shared column list, untouched.
    changed = [ln for ln, sh in zip(cols.splitlines(), _JOB_COLUMNS.splitlines(), strict=True) if ln != sh]
    assert len(changed) == 3


class RecordingPool:
    """Captures every statement JobQueries issues and answers with empty or
    zero results, so the SQL can be asserted without a database."""

    def __init__(self, tag: str = "INSERT 0 1") -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.tag = tag

    async def fetchval(self, sql, *args):
        self.calls.append((sql, args))
        return 0

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        if "AS total_jobs" in sql:
            return {"total_jobs": 0, "new_jobs": 0, "applied_jobs": 0, "total_sites": 0,
                    "jobs_last7": 0, "jobs_last30": 0, "last_scrape_at": None}
        return None

    async def execute(self, sql, *args):
        self.calls.append((sql, args))
        return self.tag


def _queries(pool: RecordingPool) -> JobQueries:
    q = JobQueries()
    q.pool = pool  # type: ignore[assignment]
    return q


async def test_reads_without_a_user_never_join_the_overlay():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(status="new"), 1, 50)
    await q.get_job(7)
    await q.job_facets()
    assert pool.calls and all("r3_user_jobs" not in sql and "uj." not in sql for sql, _ in pool.calls)
    # The portal-wide overview counts the jobs any account marked applied.
    await q.overview()
    sql, args = pool.calls[-1]
    assert "uj." not in sql and "count(DISTINCT job_id) FROM r3_user_jobs WHERE status = 'applied'" in sql
    assert args == ()


async def test_reads_with_a_user_join_the_overlay_for_that_user():
    pool = RecordingPool()
    q = _queries(pool)
    await q.list_jobs(db.JobFilter(status="skipped"), 1, 50, user_id=9001)
    count_sql, count_args = pool.calls[0]
    assert USER_JOIN_SQL in count_sql and f"{USER_STATUS_SQL} = $2" in count_sql
    assert count_args == (9001, "skipped")
    await q.get_job(7, user_id=9001)
    sql, args = pool.calls[-1]
    assert USER_JOIN_SQL in sql and "WHERE j.id = $2" in sql and args == (9001, 7)
    await q.job_facets(user_id=9001)
    sql, args = pool.calls[-1]
    assert f"{USER_STATUS_SQL} AS value" in sql and USER_JOIN_SQL in sql and args == (9001,)
    await q.overview(user_id=9001)
    sql, args = pool.calls[-1]
    assert sql.count(USER_JOIN_SQL) == 2 and args == (9001,)


async def test_user_writes_upsert_the_overlay_and_never_update_shared_rows():
    pool = RecordingPool()
    q = _queries(pool)
    await q.set_user_job_status(9001, 7, "applied")
    await q.set_user_job_notes(9001, 7, "  call back  ")
    for sql, _ in pool.calls:
        assert "INSERT INTO r3_user_jobs" in sql and "UPDATE scraped_jobs" not in sql
    status_sql, status_args = pool.calls[0]
    assert "ON CONFLICT (user_id, job_id) DO UPDATE" in status_sql and status_args == (9001, 7, "applied")
    assert "COALESCE(uj.applied_at, now())" in status_sql and "ELSE NULL" in status_sql
    assert pool.calls[1][1] == (9001, 7, "call back")


async def test_user_writes_reject_unknown_status_and_missing_job():
    with pytest.raises(ValueError):
        await _queries(RecordingPool()).set_user_job_status(1, 7, "bogus")
    with pytest.raises(LookupError):
        await _queries(RecordingPool(tag="INSERT 0 0")).set_user_job_status(1, 999, "skipped")
    with pytest.raises(LookupError):
        await _queries(RecordingPool(tag="INSERT 0 0")).set_user_job_notes(1, 999, "x")


# --- Routes over an in-memory overlay --------------------------------------

EXPIRED = JOB.model_copy(update={"id": 8, "job_url": "https://www.boards.example/jobs/8",
                                 "position": "Gone Engineer", "status": "expired"})


class OverlayDB(StubDB):
    """StubDB whose job reads and writes follow the r3_user_jobs semantics:
    shared jobs plus a per-user overlay keyed by (user_id, job_id)."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = {JOB.id: JOB, EXPIRED.id: EXPIRED}
        self.overlay: dict[tuple[int, int], dict] = {}
        self.label_pool = list(self.shared.values())

    def _view(self, job: db.Job, user_id: int | None) -> db.Job:
        if user_id is None:
            return job
        row = self.overlay.get((user_id, job.id))
        if row is None:
            return job.model_copy(update={"status": "expired" if job.status == "expired" else "new",
                                          "notes": "", "applied_at": None})
        return job.model_copy(update=row)

    async def list_jobs(self, f, page, size, *, user_id=None):
        self.last_filter, self.last_user_id = f, user_id
        rows = [self._view(j, user_id) for j in self.shared.values()]
        rows = [j for j in rows if not f.status or j.status == f.status]
        return rows, db.Page(number=1, size=size, total=len(rows), pages=1, has_prev=False, has_next=False)

    async def get_job(self, job_id, *, user_id=None):
        job = self.shared.get(job_id)
        return self._view(job, user_id) if job else None

    async def job_facets(self, *, user_id=None):
        facets = await super().job_facets(user_id=user_id)
        counts: dict[str, int] = {}
        for j in self.shared.values():
            st = self._view(j, user_id).status
            counts[st] = counts.get(st, 0) + 1
        facets.statuses = [db.FacetValue(value=k, count=v) for k, v in counts.items()]
        return facets

    async def set_user_job_status(self, user_id, job_id, status):
        await super().set_user_job_status(user_id, job_id, status)
        current = self._view(self.shared[job_id], user_id)
        self.overlay[(user_id, job_id)] = {
            "status": status, "notes": current.notes,
            "applied_at": datetime.now(timezone.utc) if status == "applied" else None,
        }

    async def set_user_job_notes(self, user_id, job_id, notes):
        await super().set_user_job_notes(user_id, job_id, notes)
        current = self._view(self.shared[job_id], user_id)
        self.overlay[(user_id, job_id)] = {"status": current.status, "notes": notes.strip(),
                                           "applied_at": current.applied_at}


ALICE = User(id=1, account_name="alice")
BOB = User(id=2, account_name="bob")


@pytest.fixture
async def site():
    """The app, its overlay stub and a client; ``as_user`` switches accounts."""
    stub = OverlayDB()
    app = create_app(database=stub)

    def as_user(user: User) -> None:
        app.dependency_overrides[current_user] = lambda: user

    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c, stub, as_user


async def test_one_users_triage_is_invisible_to_another(site):
    client, stub, as_user = site
    as_user(ALICE)
    assert (await client.post("/jobs/7/status", data={"status": "skipped"})).status_code == 200
    assert (await client.post("/jobs/7/notes", data={"notes": "recruiter spam"})).status_code == 303
    alice_page = (await client.get("/jobs/7")).text
    assert 'class="pill skipped"' in alice_page and "recruiter spam" in alice_page

    as_user(BOB)
    bob_page = (await client.get("/jobs/7")).text
    assert 'class="pill new"' in bob_page and "recruiter spam" not in bob_page
    bob_card = next(j for j in (await client.get("/api/jobs")).json()["jobs"] if j["id"] == 7)
    assert bob_card["status"] == "new"
    # The writes went to Alice's overlay only; the shared row is untouched.
    assert stub.writes == [("user_status", 1, 7, "skipped"), ("user_notes", 1, 7, "recruiter spam")]
    assert stub.shared[7].status == "new" and stub.shared[7].notes == ""


async def test_status_filter_and_counts_are_per_user(site):
    client, stub, as_user = site
    as_user(ALICE)
    await client.post("/jobs/7/status", data={"status": "skipped"})
    assert (await client.get("/api/jobs?status=skipped")).json()["total"] == 1
    assert stub.last_user_id == ALICE.id
    assert "skipped (1)" in (await client.get("/jobs")).text

    as_user(BOB)
    assert (await client.get("/api/jobs?status=skipped")).json()["total"] == 0
    assert (await client.get("/api/jobs?status=new")).json()["total"] == 1
    page = (await client.get("/jobs")).text
    assert "skipped (" not in page and "new (1)" in page
    assert stub.last_user_id == BOB.id


async def test_shared_expired_shows_through_without_an_overlay_row(site):
    client, _, as_user = site
    as_user(BOB)
    assert 'class="pill expired"' in (await client.get("/jobs/8")).text
    assert (await client.get("/api/jobs?status=expired")).json()["total"] == 1
    # A note on it keeps it expired rather than resurrecting it as "new".
    await client.post("/jobs/8/notes", data={"notes": "missed it"})
    assert 'class="pill expired"' in (await client.get("/jobs/8")).text


async def test_signed_out_requests_use_the_local_account(site):
    client, stub, _ = site
    await client.post("/jobs/7/status", data={"status": "applied"})
    assert stub.writes == [("user_status", 0, 7, "applied")]
