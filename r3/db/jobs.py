"""Queries over ``scraped_jobs``: the jobs list, detail, triage writes, the
filter facets and the overview counters."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from urllib.parse import urlencode

from pydantic import BaseModel, Field

from ._base import QueryBase, WhereBuilder

# Postgres dates arrive as datetime.date and timestamps as datetime; the
# templates format either, so the row models accept both.
Timestamp = datetime | date


class Job(BaseModel):
    """A row of scraped_jobs with the viewer's own triage (status, notes,
    applied_at) from r3_user_jobs."""

    id: int
    site_url: str
    site_name: str
    job_url: str
    apply_url: str = ""
    position: str
    company: str = ""
    location: str = ""
    posted_date: Timestamp | None = None
    scraped_at: datetime
    role_category: str = ""
    employment: str = ""
    salary: str = ""
    salary_min: int | None = None
    salary_max: int | None = None
    status: str
    notes: str = ""
    tech_tags: list[str] = Field(default_factory=list)
    location_kind: str = ""
    description: str = ""
    applied_at: datetime | None = None

    last_validation: str = ""
    last_validated_at: datetime | None = None
    verification_status: str = ""
    # False once the posting is gone (last_validation = 'expired'). A fact
    # about the job, whatever the viewer's own triage says.
    live: bool = True

    # Resume match, set only when list_jobs is given the user's skills:
    # how many of the posting's tech_tags the resume shows, out of how many.
    match_count: int | None = None
    match_total: int | None = None


class JobFilter(BaseModel):
    """The set of user-selectable constraints on the jobs list."""

    search: str = ""  # position, company, or description
    status: str = ""
    role_category: str = ""
    location_kind: str = ""
    site_url: str = ""
    tech_tag: str = ""
    employment: str = ""
    min_salary: int = 0
    days: int = 0  # scraped within the last N days
    sort: str = ""

    # role_categories matches any of several categories at once. Takes
    # precedence over role_category.
    role_categories: list[str] = Field(default_factory=list)
    # employment_family matches a group of employment spellings at once
    # (see EMPLOYMENT_FAMILIES). Takes precedence over employment.
    employment_family: str = ""
    # remote matches any of the three remote location kinds.
    remote: bool = False
    # has_salary keeps only postings that state pay.
    has_salary: bool = False

    def query_string(self) -> str:
        """Render the filter back into a /jobs query string, so a preset link
        and the count behind it always describe the same search."""
        pairs: list[tuple[str, str]] = []

        def set_(key: str, val: str) -> None:
            if val:
                pairs.append((key, val))

        set_("q", self.search)
        set_("status", self.status)
        if self.role_categories:
            pairs.extend(("role", rc) for rc in self.role_categories)
        else:
            set_("role", self.role_category)
        set_("location", self.location_kind)
        set_("site", self.site_url)
        set_("tech", self.tech_tag)
        set_("employment", self.employment)
        set_("employment_family", self.employment_family)
        sort = self.sort
        if self.min_salary > 0:
            pairs.append(("min_salary", str(self.min_salary)))
            # A salary floor is meaningless unsorted, so it forces the sort.
            sort = "salary"
        set_("sort", sort)
        if self.days > 0:
            pairs.append(("days", str(self.days)))
        if self.remote:
            pairs.append(("remote", "1"))
        if self.has_salary:
            pairs.append(("has_salary", "1"))
        return "?" + urlencode(pairs)


class Page(BaseModel):
    """One slice of a result set."""

    number: int
    size: int
    total: int
    pages: int
    has_prev: bool
    has_next: bool


class FacetValue(BaseModel):
    """One dropdown option with its row count."""

    value: str
    count: int


class Facets(BaseModel):
    """Distinct values used to populate the jobs filter dropdowns."""

    statuses: list[FacetValue] = Field(default_factory=list)
    role_categories: list[FacetValue] = Field(default_factory=list)
    location_kinds: list[FacetValue] = Field(default_factory=list)
    employments: list[FacetValue] = Field(default_factory=list)
    tech_tags: list[FacetValue] = Field(default_factory=list)


class Stats(BaseModel):
    """Headline numbers on the overview page."""

    total_jobs: int
    new_jobs: int
    applied_jobs: int
    total_sites: int
    jobs_last7: int
    jobs_last30: int
    last_scrape_at: datetime | None = None


class SiteOption(BaseModel):
    """A lightweight site entry for the jobs page site dropdown."""

    url: str
    name: str
    count: int


# EMPLOYMENT_FAMILIES collapses the messy employment vocabulary into the four
# groups worth searching by. Postings say both "fulltime" and "Full-time",
# and treat c2c as its own contract flavor, so an exact-match filter would
# silently hide most of the postings a user asked for.
EMPLOYMENT_FAMILIES: dict[str, list[str]] = {
    "contract": ["contract", "Contract", "c2c", "Contract (12 Months)"],
    "fulltime": ["fulltime", "Full-time"],
    "parttime": ["parttime", "Part-time"],
    "internship": ["internship"],
}

# ALLOWED_STATUSES are the values an account may give a job (r3_user_jobs.status).
ALLOWED_STATUSES = frozenset({"new", "applied", "skipped", "expired"})

JOBS_SORT_SQL: dict[str, str] = {
    "newest": "j.scraped_at DESC",
    "oldest": "j.scraped_at ASC",
    "posted": "j.posted_date DESC NULLS LAST",
    "company": "j.company ASC NULLS LAST, j.position ASC",
    "position": "j.position ASC",
    "salary": "j.salary_max DESC NULLS LAST",
    "salary_asc": "j.salary_min ASC NULLS LAST",
    "site": "s.name ASC, j.scraped_at DESC",
}

JOBS_SORT_DEFAULT = "j.scraped_at DESC"

# "match" ranks by fit with the user's resume. It needs the skills bound as a
# parameter, so it is not in JOBS_SORT_SQL: list_jobs uses it only when given
# skills, and otherwise falls back to the default like any unknown sort.
# Live postings come first, then the title fit with the roles the resume
# names (when it names any), then the overlap with its skills; ties go to
# the posting the overlap covers more of, then the newest.
JOBS_SORT_MATCH = "match"
MATCH_SORT_SQL = (
    "m.match_count DESC,"
    " m.match_count::float / NULLIF(cardinality(j.tech_tags), 0) DESC NULLS LAST,"
    " j.scraped_at DESC"
)
# LIVE_SQL is whether the posting is still up. An expired posting is gone, so
# it never outranks a live one whatever the user's own triage says.
LIVE_SQL = "(COALESCE(j.last_validation, '') <> 'expired')"
MATCH_LIVE_SQL = f"{LIVE_SQL} DESC"


def title_fit_sql(target: int, recent: int) -> str:
    """2 when the job's title matches a pattern of the resume's target role
    (bound as ``$target``), 1 for one of its other recent roles (``$recent``),
    else 0. The patterns come from :func:`r3.resume.roles.title_patterns`."""
    return (f"CASE WHEN j.position ~* ANY(${target}::text[]) THEN 2"
            f" WHEN j.position ~* ANY(${recent}::text[]) THEN 1 ELSE 0 END")


def match_order(title_fit: str = "") -> str:
    """The ORDER BY of the match sort, with the title fit when there is one."""
    fit = f"{title_fit} DESC, " if title_fit else ""
    return f"{MATCH_LIVE_SQL}, {fit}{MATCH_SORT_SQL}"


def match_sql(n: int) -> tuple[str, str]:
    """The extra select columns and the LATERAL join that count a job's
    tech_tags found in the skills array bound as ``$n``."""
    cols = ",\n    m.match_count, cardinality(COALESCE(j.tech_tags, '{}')) AS match_total"
    join = (
        "CROSS JOIN LATERAL (SELECT count(DISTINCT t)::int AS match_count"
        f" FROM unnest(j.tech_tags) AS t WHERE t = ANY(${n}::text[])) m"
    )
    return cols, join

# r3_user_jobs holds each account's triage of a job: status, notes and when
# they applied. scraped_jobs carries only facts about the posting. No foreign
# key to scraped_jobs on purpose: old postings get deleted, and an account's
# history must not block that.
USER_JOBS_DDL = """
CREATE TABLE IF NOT EXISTS r3_user_jobs (
    user_id    bigint      NOT NULL,
    job_id     bigint      NOT NULL,
    status     text        NOT NULL DEFAULT 'new',
    notes      text        NOT NULL DEFAULT '',
    applied_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, job_id)
)"""

# SHARED_STATUS_SQL is a job's status before anyone triaged it: "new", or
# "expired" once the posting is gone, since that is a fact about the job.
SHARED_STATUS_SQL = f"CASE WHEN {LIVE_SQL} THEN 'new' ELSE 'expired' END"
# USER_STATUS_SQL is a job's effective status for one user: their own choice,
# else the shared status.
USER_STATUS_SQL = f"COALESCE(uj.status, {SHARED_STATUS_SQL})"

# USER_JOIN_SQL attaches the overlay row for the user bound to $1. The primary
# key makes it at most one row per job, so counts are unaffected.
USER_JOIN_SQL = "LEFT JOIN r3_user_jobs uj ON uj.job_id = j.id AND uj.user_id = $1"


def overlay_join(user_id: int | None) -> str:
    """The overlay join for a per-user query, or nothing for the shared view."""
    return "" if user_id is None else USER_JOIN_SQL


def build_job_where(f: JobFilter, user_id: int | None = None) -> tuple[str, list[Any]]:
    """Render the shared WHERE clause and its arguments. Both list_jobs and
    the count use it so the count always matches the page.

    With a ``user_id`` the status filter reads that user's overlay, and the id
    is bound as ``$1`` for :data:`USER_JOIN_SQL`, which the caller must then
    join. Without one the status filter is the shared status (new or
    expired)."""
    w = WhereBuilder()
    if user_id is not None:
        # Reserve $1 before any filter value so the join's placeholder is fixed.
        w.args.append(user_id)

    if s := f.search.strip()[:200]:
        s = s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        w.add("(j.position ILIKE ${n} ESCAPE '\\' OR j.company ILIKE ${n} ESCAPE '\\'"
              " OR j.description ILIKE ${n} ESCAPE '\\')", f"%{s}%")
    if v := f.status.strip():
        w.add((SHARED_STATUS_SQL if user_id is None else USER_STATUS_SQL) + " = ${n}", v)
    # role_categories (plural) lets one search span sibling categories, e.g.
    # DevOps + SRE. It supersedes the single-value form when set.
    if f.role_categories:
        w.add("j.role_category = ANY(${n})", list(f.role_categories))
    elif v := f.role_category.strip():
        w.add("j.role_category = ${n}", v)
    if v := f.location_kind.strip():
        w.add("j.location_kind = ${n}", v)
    if v := f.site_url.strip():
        w.add("j.site_url = ${n}", v)
    # The family filter supersedes the exact one: a preset asking for
    # "contract" means every contract spelling, not the literal string.
    if fam := f.employment_family.strip():
        if vals := EMPLOYMENT_FAMILIES.get(fam):
            w.add("j.employment = ANY(${n})", vals)
    elif v := f.employment.strip():
        w.add("j.employment = ${n}", v)
    if f.remote:
        w.raw("j.location_kind IN ('remote-us', 'remote-ww', 'remote-ca')")
    if f.has_salary:
        w.raw("j.salary IS NOT NULL AND j.salary <> ''")
    if v := f.tech_tag.strip():
        # GIN index on tech_tags makes the containment operator the fast path.
        w.add("j.tech_tags @> ARRAY[${n}]::text[]", v)
    if f.min_salary > 0:
        # salary_max is inconsistently scaled upstream: most yearly rows store
        # dollars, but ~90 store cents (56600000 meaning $566,000). No genuine
        # annual figure in this data reaches $1M, so a yearly value at or above
        # that is cents with near-certainty and is scaled down before compare.
        # The filter is restricted to yearly rows — comparing an annual floor
        # against an hourly rate would silently drop every hourly posting.
        w.add(
            """(j.salary_period IN ('year', 'annual')
              AND (CASE WHEN j.salary_max >= 1000000 THEN j.salary_max / 100
                        ELSE j.salary_max END) >= ${n})""",
            f.min_salary,
        )
    if f.days > 0:
        w.add("j.scraped_at >= now() - make_interval(days => ${n})", f.days)

    return w.clause(), w.args


# Columns shared by the list and detail queries; the detail adds description.
# Without a viewer nobody's triage applies: the shared status, no notes.
_JOB_COLUMNS = f"""
    j.id, j.site_url, s.name AS site_name, j.job_url,
    COALESCE(j.apply_url, '') AS apply_url,
    j.position, COALESCE(j.company, '') AS company,
    COALESCE(j.location, '') AS location,
    j.posted_date, j.scraped_at,
    COALESCE(j.role_category, '') AS role_category,
    COALESCE(j.employment, '') AS employment,
    COALESCE(j.salary, '') AS salary,
    j.salary_min, j.salary_max, {SHARED_STATUS_SQL} AS status,
    '' AS notes,
    COALESCE(j.tech_tags, '{{}}') AS tech_tags,
    COALESCE(j.location_kind, '') AS location_kind,
    NULL::timestamptz AS applied_at,
    COALESCE(j.last_validation, '') AS last_validation,
    j.last_validated_at,
    COALESCE(j.verification_status, '') AS verification_status,
    {LIVE_SQL} AS live"""

# The same columns with the triage fields read from the user's overlay.
_USER_JOB_COLUMNS = (
    _JOB_COLUMNS.replace(f"{SHARED_STATUS_SQL} AS status,", f"{USER_STATUS_SQL} AS status,")
    .replace("'' AS notes", "COALESCE(uj.notes, '') AS notes")
    .replace("NULL::timestamptz AS applied_at,", "uj.applied_at AS applied_at,")
)


def job_columns(user_id: int | None) -> str:
    """The list/detail column list for the shared view or one user's view."""
    return _JOB_COLUMNS if user_id is None else _USER_JOB_COLUMNS


class JobQueries(QueryBase):
    async def ensure_user_jobs_table(self) -> None:
        """Create the per-user triage overlay if it does not exist yet."""
        await self.pool.execute(USER_JOBS_DDL)

    async def list_jobs(
        self, f: JobFilter, page: int, size: int, *, user_id: int | None = None,
        skills: list[str] | None = None, titles: tuple[list[str], list[str]] | None = None,
    ) -> tuple[list[Job], Page]:
        """Return one page of jobs plus the pagination state. With a
        ``user_id`` status, notes and applied_at are that user's own. With
        ``skills`` (the user's resume skills) each job carries match_count and
        match_total, and ``sort="match"`` ranks by them. ``titles`` are the
        (target, recent) title patterns of the roles the resume names, which
        that sort ranks by before the skills."""
        if page < 1:
            page = 1
        if size < 1 or size > 200:
            size = 50

        clause, args = build_job_where(f, user_id)
        join = overlay_join(user_id)

        total: int = await self.pool.fetchval(
            f"""SELECT count(*) FROM scraped_jobs j
                JOIN job_sites s ON s.url = j.site_url {join} {clause}""",
            *args,
        )

        pages = (total + size - 1) // size
        if 0 < pages < page:
            page = pages

        order = JOBS_SORT_SQL.get(f.sort, JOBS_SORT_DEFAULT)
        # With the user's resume skills each row also carries its match, and
        # the "match" sort becomes available. The skills are bound after the
        # filter values, so the count above is untouched.
        cols = job_columns(user_id)
        if skills:
            args = [*args, list(skills)]
            match_cols, match_join = match_sql(len(args))
            cols += match_cols
            join += f"\n            {match_join}"
            if f.sort == JOBS_SORT_MATCH:
                fit = ""
                if titles and titles[0]:
                    args = [*args, list(titles[0]), list(titles[1])]
                    fit = title_fit_sql(len(args) - 1, len(args))
                order = match_order(fit)

        # Description is deliberately excluded here — it is large and the list
        # view only needs it on the detail page.
        q = f"""
            SELECT {cols}
            FROM scraped_jobs j
            JOIN job_sites s ON s.url = j.site_url
            {join}
            {clause}
            ORDER BY {order}
            LIMIT ${len(args) + 1} OFFSET ${len(args) + 2}"""

        rows = await self.pool.fetch(q, *args, size, (page - 1) * size)
        jobs = [Job(**dict(r)) for r in rows]

        return jobs, Page(
            number=page,
            size=size,
            total=total,
            pages=pages,
            has_prev=page > 1,
            has_next=page < pages,
        )

    async def get_job(self, job_id: int, *, user_id: int | None = None) -> Job | None:
        """Load one job with its full description; None when it doesn't exist.
        With a ``user_id`` the triage fields are that user's own."""
        if user_id is None:
            row = await self.pool.fetchrow(
                f"""
                SELECT {_JOB_COLUMNS}, COALESCE(j.description, '') AS description
                FROM scraped_jobs j
                JOIN job_sites s ON s.url = j.site_url
                WHERE j.id = $1""",
                job_id,
            )
        else:
            row = await self.pool.fetchrow(
                f"""
                SELECT {_USER_JOB_COLUMNS}, COALESCE(j.description, '') AS description
                FROM scraped_jobs j
                JOIN job_sites s ON s.url = j.site_url
                {USER_JOIN_SQL}
                WHERE j.id = $2""",
                user_id,
                job_id,
            )
        return Job(**dict(row)) if row else None

    # The writes touch only r3_user_jobs. The INSERT ... SELECT FROM
    # scraped_jobs is the existence check: a job that is not there inserts
    # nothing, which is reported as not found.

    async def set_user_job_status(self, user_id: int, job_id: int, status: str) -> None:
        """Set one user's status on a job. Moving to "applied" stamps
        applied_at; moving away from it clears the stamp."""
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unknown job status {status!r}")
        tag = await self.pool.execute(
            """
            INSERT INTO r3_user_jobs AS uj (user_id, job_id, status, applied_at)
            SELECT $1, j.id, $3::text, CASE WHEN $3::text = 'applied' THEN now() END
            FROM scraped_jobs j WHERE j.id = $2
            ON CONFLICT (user_id, job_id) DO UPDATE
            SET status = EXCLUDED.status,
                applied_at = CASE WHEN EXCLUDED.status = 'applied' THEN COALESCE(uj.applied_at, now())
                                  ELSE NULL END,
                updated_at = now()""",
            user_id,
            job_id,
            status,
        )
        if tag == "INSERT 0 0":
            raise LookupError(f"job not found: {job_id}")

    async def set_user_job_notes(self, user_id: int, job_id: int, notes: str) -> None:
        """Write one user's note on a job. A first write seeds the overlay row
        with the status the user already sees, so a note on a shared-expired
        job does not turn it back into "new"."""
        if len(notes) > 10_000:
            raise ValueError("notes are limited to 10000 characters")
        tag = await self.pool.execute(
            f"""
            INSERT INTO r3_user_jobs AS uj (user_id, job_id, status, notes)
            SELECT $1, j.id, {SHARED_STATUS_SQL}, $3
            FROM scraped_jobs j WHERE j.id = $2
            ON CONFLICT (user_id, job_id) DO UPDATE
            SET notes = EXCLUDED.notes, updated_at = now()""",
            user_id,
            job_id,
            notes.strip(),
        )
        if tag == "INSERT 0 0":
            raise LookupError(f"job not found: {job_id}")

    async def job_facets(self, *, user_id: int | None = None) -> Facets:
        """Collect the distinct filter values in a single round trip. With a
        ``user_id`` the status counts are that user's own."""
        f = Facets()
        if user_id is None:
            status_sql = f"SELECT 'status' AS kind, {SHARED_STATUS_SQL} AS value, count(*) AS n FROM scraped_jobs j GROUP BY 2"
            args: tuple[Any, ...] = ()
        else:
            status_sql = (
                f"SELECT 'status' AS kind, {USER_STATUS_SQL} AS value, count(*) AS n"
                f" FROM scraped_jobs j {USER_JOIN_SQL} GROUP BY 2"
            )
            args = (user_id,)
        # One UNION ALL keeps this to a single query rather than five.
        rows = await self.pool.fetch(
            f"""
            {status_sql}
            UNION ALL
            SELECT 'role', role_category, count(*) FROM scraped_jobs
              WHERE role_category IS NOT NULL AND role_category <> '' GROUP BY 2
            UNION ALL
            SELECT 'location', location_kind, count(*) FROM scraped_jobs
              WHERE location_kind IS NOT NULL AND location_kind <> '' GROUP BY 2
            UNION ALL
            SELECT 'employment', employment, count(*) FROM scraped_jobs
              WHERE employment IS NOT NULL AND employment <> '' GROUP BY 2
            UNION ALL
            SELECT 'tech', tag, count(*) FROM scraped_jobs, unnest(tech_tags) AS tag GROUP BY 2
            ORDER BY 1, 3 DESC""",
            *args,
        )
        for r in rows:
            fv = FacetValue(value=r["value"], count=r["n"])
            match r["kind"]:
                case "status":
                    f.statuses.append(fv)
                case "role":
                    f.role_categories.append(fv)
                case "location":
                    f.location_kinds.append(fv)
                case "employment":
                    f.employments.append(fv)
                case "tech":
                    # The tag vocabulary is long-tailed; the dropdown only needs
                    # the ones that actually narrow a search.
                    if len(f.tech_tags) < 60:
                        f.tech_tags.append(fv)
        return f

    async def tech_vocabulary(self) -> list[str]:
        """Every distinct tech tag in the corpus: the vocabulary a resume's
        skills are recognised from. job_facets caps its tag list for the
        dropdown, so this is its own query; callers cache it."""
        rows = await self.pool.fetch(
            "SELECT DISTINCT tag FROM scraped_jobs, unnest(tech_tags) AS tag"
            " WHERE tag IS NOT NULL AND tag <> '' ORDER BY 1"
        )
        return [r["tag"] for r in rows]

    async def overview(self, *, user_id: int | None = None) -> Stats:
        """Load the dashboard counters. With a ``user_id`` the new/applied
        counts are that user's own. Without one they describe the shared
        corpus: every live job is new, and applied counts the jobs any account
        marked applied."""
        if user_id is None:
            new_sql = f"(SELECT count(*) FROM scraped_jobs j WHERE {LIVE_SQL})"
            applied_sql = "(SELECT count(DISTINCT job_id) FROM r3_user_jobs WHERE status = 'applied')"
            args: tuple[Any, ...] = ()
        else:
            new_sql = f"(SELECT count(*) FROM scraped_jobs j {USER_JOIN_SQL} WHERE {USER_STATUS_SQL} = 'new')"
            applied_sql = f"(SELECT count(*) FROM scraped_jobs j {USER_JOIN_SQL} WHERE {USER_STATUS_SQL} = 'applied')"
            args = (user_id,)
        r = await self.pool.fetchrow(
            f"""
            SELECT
              (SELECT count(*) FROM scraped_jobs)                          AS total_jobs,
              {new_sql}     AS new_jobs,
              {applied_sql} AS applied_jobs,
              (SELECT count(*) FROM job_sites)                             AS total_sites,
              (SELECT count(*) FROM scraped_jobs
                 WHERE scraped_at >= now() - interval '7 days')            AS jobs_last7,
              (SELECT count(*) FROM scraped_jobs
                 WHERE scraped_at >= now() - interval '30 days')           AS jobs_last30,
              (SELECT max(scraped_at) FROM scraped_jobs)                   AS last_scrape_at""",
            *args,
        )
        return Stats(**dict(r))

    async def sites_with_jobs(self) -> list[SiteOption]:
        """List sites that actually have jobs, for the filter dropdown."""
        rows = await self.pool.fetch(
            """
            SELECT s.url, s.name, count(j.id) AS count
            FROM job_sites s
            JOIN scraped_jobs j ON j.site_url = s.url
            GROUP BY s.url, s.name
            ORDER BY count DESC, s.name"""
        )
        return [SiteOption(**dict(r)) for r in rows]
