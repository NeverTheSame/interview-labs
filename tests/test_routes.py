"""Routing and rendering end to end, against a stub database. Covers the
HTMX partial/full split, the single-row swap targets and the error paths."""

from datetime import datetime, timedelta, timezone

import pytest
from httpx import ASGITransport, AsyncClient

from r3 import db
from r3.web import create_app

NOW = datetime.now(timezone.utc)

JOB = db.Job(
    id=7, site_url="https://www.boards.example/jobs", site_name="Example Boards",
    job_url="https://www.boards.example/jobs/7", position="Senior ML Engineer", company="Acme AI",
    location="Remote, US", scraped_at=NOW - timedelta(hours=3), role_category="AI/ML",
    employment="fulltime", salary="USD 180,000 - 220,000 per year", salary_min=180000, salary_max=220000,
    status="new", tech_tags=["python", "kubernetes"], location_kind="remote-us",
    description="Build models.\n\nShip them.",
)
SITE_URL, SITE_NAME = "https://www.boards.example/jobs", "Example Boards"


class StubDB:
    """Answers every query with canned rows and records the writes."""

    def __init__(self) -> None:
        self.writes: list[tuple] = []
        self.ping_ok = True
        self.pings = 0
        self.last_filter: db.JobFilter | None = None
        self.last_page: tuple[int, int] | None = None
        # The postings the labeling page walks (tests add more); get_job reads it too.
        self.label_pool: list[db.Job] = [JOB]
        self.last_seen: list[int] = []
        # The user_id the job reads were scoped to (None = the shared view).
        self.last_user_id: int | None = None
        self.summaries: dict[int, db.JobSummary] = {}

    async def ping(self):
        self.pings += 1
        if not self.ping_ok:
            raise ConnectionError("down")

    async def close(self):
        self.closed = True

    async def overview(self, *, user_id=None):
        return db.Stats(total_jobs=100, new_jobs=40, applied_jobs=5, total_sites=4,
                        jobs_last7=10, jobs_last30=30, last_scrape_at=NOW - timedelta(hours=1))

    async def job_facets(self, *, user_id=None):
        self.last_user_id = user_id
        return db.Facets(statuses=[db.FacetValue(value="new", count=40)],
                         role_categories=[db.FacetValue(value="AI/ML", count=20)],
                         location_kinds=[db.FacetValue(value="remote-us", count=30)],
                         employments=[db.FacetValue(value="fulltime", count=50)],
                         tech_tags=[db.FacetValue(value="python", count=60)])

    async def sites_with_jobs(self):
        return [db.SiteOption(url=SITE_URL, name=SITE_NAME, count=12)]

    async def list_jobs(self, f, page, size, *, user_id=None):
        self.last_filter, self.last_page, self.last_user_id = f, (page, size), user_id
        return [JOB], db.Page(number=2, size=size, total=101, pages=3, has_prev=True, has_next=True)

    async def get_job(self, job_id, *, user_id=None):
        return next((job for job in self.label_pool if job.id == job_id), None)

    # The job overview cache (r3_job_summaries), in memory.
    async def get_job_summary(self, job_id):
        return self.summaries.get(job_id)

    async def save_job_summary(self, job_id, summary, model):
        self.summaries[job_id] = db.JobSummary(job_id=job_id, summary=summary, model=model, created_at=NOW)

    # The job match reads these for regular accounts; no resume by default.
    async def primary_resume(self, user_id):
        return None

    async def tech_vocabulary(self):
        return []

    async def set_user_job_status(self, user_id, job_id, status):
        if status not in db.ALLOWED_STATUSES:
            raise ValueError("unknown job status")
        self.writes.append(("user_status", user_id, job_id, status))

    async def set_user_job_notes(self, user_id, job_id, notes):
        self.writes.append(("user_notes", user_id, job_id, notes))

    async def scout_presets(self, scope, *, user_id=None):
        link = "?status=new&role=AI%2FML"
        return [db.PresetGroup(title="By role", items=[db.Preset(label="AI/ML", query=link, count=20)])], 40

    async def scout_combos(self, scope, *, user_id=None):
        return [db.Combo(label="Fresh remote AI/ML", hint="h", query="?status=new", count=3)]

    # --- labels (r3/db/labels.py) ---

    async def ensure_labels_tables(self):
        if not hasattr(self, "labels"):
            self.labels: dict[tuple[int, str], db.Label] = {}
            self.label_history: list[db.Label] = []

    async def get_label(self, job_id, rubric_version):
        await self.ensure_labels_tables()
        return self.labels.get((job_id, rubric_version))

    async def save_label(self, job_id, rubric_version, answers, note, *, labeler, session_id, seconds_spent):
        await self.ensure_labels_tables()
        previous = self.labels.get((job_id, rubric_version))
        if previous is not None:
            self.label_history.append(previous)
        label = db.Label(id=previous.id if previous else len(self.labels) + 1, job_id=job_id,
                         rubric_version=rubric_version, answers=answers, note=note,
                         labeled_at=datetime.now(timezone.utc), labeler=labeler, session_id=session_id,
                         seconds_spent=seconds_spent)
        self.labels[(job_id, rubric_version)] = label
        self.writes.append(("label", job_id, answers.get("relevant")))
        return label, previous is not None

    def _label_queue(self, rubric_version, seen):
        seen = set(seen)
        return [j for j in self.label_pool if (j.id, rubric_version) not in self.labels and j.id not in seen]

    async def next_job_to_label(self, f, rubric_version, seen):
        await self.ensure_labels_tables()
        self.last_filter, self.last_seen = f, list(seen)
        queue = self._label_queue(rubric_version, seen)
        return queue[0] if queue else None

    async def label_counts(self, f, rubric_version, seen, since):
        await self.ensure_labels_tables()
        mine = [l for l in self.labels.values() if l.rubric_version == rubric_version]
        return db.LabelCounts(today=sum(l.labeled_at >= since for l in mine), total=len(mine),
                              remaining=len(self._label_queue(rubric_version, seen)))

    async def export_labels(self, rubric_version):
        await self.ensure_labels_tables()
        out = []
        for label in sorted(self.labels.values(), key=lambda l: (l.labeled_at, l.id)):
            if label.rubric_version != rubric_version:
                continue
            job = next((j for j in self.label_pool if j.id == label.job_id), None)
            snapshot = db.JobSnapshot(**job.model_dump()) if job else None
            out.append(db.LabelExport(rubric_version=rubric_version, job_id=label.job_id, job=snapshot,
                                      answers=label.answers, note=label.note, labeled_at=label.labeled_at,
                                      labeler=label.labeler, session_id=label.session_id,
                                      seconds_spent=label.seconds_spent))
        return out

    # --- portal settings (r3/db/settings.py) ---

    async def ensure_settings_table(self):
        if not hasattr(self, "settings_rows"):
            self.settings_rows: dict[str, tuple[str, bool]] = {}

    async def load_settings(self):
        return {k: v for k, (v, _secret) in getattr(self, "settings_rows", {}).items()}

    async def save_setting(self, key, value, secret=False):
        await self.ensure_settings_table()
        self.settings_rows[key] = (value, secret)
        self.writes.append(("setting", key))

    async def delete_setting(self, key):
        await self.ensure_settings_table()
        self.settings_rows.pop(key, None)
        self.writes.append(("setting_deleted", key))

@pytest.fixture
def stub():
    return StubDB()


@pytest.fixture
async def client(stub):
    app = create_app(database=stub)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


async def test_liveness_never_touches_the_database(client, stub):
    # A frequent health check on /healthz must never keep a scale-to-zero
    # database awake, so liveness answers without it.
    stub.ping_ok = False
    resp = await client.get("/healthz")
    assert (resp.status_code, resp.text) == (200, "OK")
    assert stub.pings == 0


async def test_readiness_checks_the_database(client, stub):
    assert (await client.get("/readyz")).text == "OK"
    stub.ping_ok = False
    assert (await client.get("/readyz")).status_code == 503
    assert stub.pings == 2


@pytest.mark.parametrize("path, needle", [
    ("/", "jobs total"),
    ("/controller", "Run the job search from here."),
    ("/scout", "Curated hunts"),
    ("/scout?scope=all", "All jobs"),
    ("/jobs", "job-filters"),
    ("/jobs/7", "Save notes"),
])
async def test_pages_render(client, path, needle):
    r = await client.get(path)
    assert r.status_code == 200, r.text[:300]
    assert needle in r.text
    assert "<!doctype html>" in r.text


async def test_controller_shows_pipelines_beside_the_chat(client):
    r = await client.get("/controller")
    body = r.text
    assert body.index('id="r3-chat"') < body.index('id="r3-panel"') < body.index('id="r3-pipelines"')
    for name in ("Jobs", "Tracking"):
        assert f'<article class="pipe" data-pipeline="{name.lower()}"' in body and f"<h3>{name}</h3>" in body
    assert "40</dd>" in body and "100</dd>" in body  # live facts from the stub


async def test_controller_renders_without_database(client, stub):
    async def down():
        raise ConnectionError("down")
    stub.overview = down
    r = await client.get("/controller")
    assert r.status_code == 200
    assert "Live numbers unavailable" in r.text
    assert 'id="r3-chat"' in r.text and 'id="r3-pipelines"' in r.text


async def test_jobs_query_maps_to_filter(client, stub):
    await client.get("/jobs?q=kafka&status=new&role=DevOps&role=SRE&remote=1&min_salary=200000&days=7&page=3&size=25")
    f = stub.last_filter
    assert f.search == "kafka" and f.status == "new"
    assert f.role_categories == ["DevOps", "SRE"] and f.role_category == "DevOps"
    assert f.remote and f.min_salary == 200000 and f.days == 7
    assert stub.last_page == (3, 25)


async def test_jobs_htmx_returns_only_the_table(client):
    r = await client.get("/jobs?page=2", headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert "<!doctype html>" not in r.text and "job-filters" not in r.text
    assert "resulthead" in r.text
    # Pager links carry the current query forward with the page overridden.
    assert 'href="/jobs?page=3"' in r.text and 'href="/jobs?page=1"' in r.text


async def test_status_action_returns_single_row(client, stub):
    r = await client.post("/jobs/7/status", data={"status": "applied"})
    assert r.status_code == 200
    assert r.text.strip().startswith('<tr id="job-7"') and "<table" not in r.text
    assert stub.writes == [("user_status", 0, 7, "applied")]


async def test_status_action_from_detail_redirects(client):
    r = await client.post("/jobs/7/status", data={"status": "skipped", "redirect": "1"})
    assert r.status_code == 303 and r.headers["location"] == "/jobs/7"


async def test_bad_status_and_missing_job(client):
    assert (await client.post("/jobs/7/status", data={"status": "bogus"})).status_code == 400
    assert (await client.get("/jobs/999")).status_code == 404


async def test_notes_redirects_back(client, stub):
    r = await client.post("/jobs/7/notes", data={"notes": "call back"})
    assert r.status_code == 303 and stub.writes == [("user_notes", 0, 7, "call back")]


async def test_readiness_answers_without_sign_in_like_liveness():
    from r3.web.auth import _OPEN_PATHS
    from tests.test_user_shell import ShellStub

    assert {"/healthz", "/readyz"} <= _OPEN_PATHS
    app = create_app(database=ShellStub(), auth=True)
    async with app.router.lifespan_context(app), AsyncClient(transport=ASGITransport(app=app),
                                                             base_url="http://test") as c:
        assert (await c.get("/readyz")).text == "OK"
        assert (await c.head("/healthz")).status_code == 200
