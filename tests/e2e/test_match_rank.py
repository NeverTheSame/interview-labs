"""The best-match sort on real Postgres: live postings first, then title fit
from the resume's roles, then tag overlap, coverage and recency. No browser:
each test inserts its rows inside a transaction it rolls back, under a site
of its own, so no other test or session ever sees them."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from r3 import db
from r3.db.jobs import USER_JOBS_DDL, JobQueries
from r3.resume.roles import RoleKeywords, role_keywords

SITE = "https://careers.rank-test.example/"
USER_ID = 9_000_001

# position, last_validation (expired: the posting is gone), tags, hours since scraped
ROWS = [
    ("Technical Writer", "new", ["python"], 0),
    ("Product Manager", "new", ["python"], 1),
    ("DevOps Engineer", "expired", ["terraform", "kubernetes", "aws"], 2),
    ("Senior DevOps Engineer", "new", ["terraform", "kubernetes", "aws", "go"], 3),
    ("Platform Engineer", "new", ["kubernetes"], 4),
    ("Linux Systems Administrator", "new", ["linux", "bash"], 5),
    ("Site Reliability Engineer", "new", ["terraform", "kubernetes", "aws"], 6),
    ("Senior SRE", "new", ["kubernetes", "prometheus"], 7),
]

DEVOPS_SKILLS = ["aws", "bash", "kubernetes", "linux", "python", "terraform"]
SRE_SKILLS = ["aws", "go", "kubernetes", "prometheus", "terraform"]


async def _rank(dsn: str, skills: list[str], roles: RoleKeywords) -> list[tuple[str, int | None]]:
    import asyncpg

    conn = await asyncpg.connect(dsn, timeout=3)
    tx = conn.transaction()
    await tx.start()
    try:
        await conn.execute(USER_JOBS_DDL)
        await conn.execute("INSERT INTO job_sites (url, name) VALUES ($1, 'Rank Test')", SITE)
        for n, (position, validation, tags, hours) in enumerate(ROWS):
            await conn.execute(
                "INSERT INTO scraped_jobs (site_url, job_url, position, company, description, last_validation,"
                " tech_tags, scraped_at) VALUES ($1, $2, $3, 'Rank Test', 'A synthetic posting for the rank test.',"
                " NULLIF($4, 'new'), $5, now() - make_interval(hours => $6))",
                SITE, f"{SITE}jobs/{n}", position, validation, tags, hours)
        q = JobQueries()
        q.pool = conn  # type: ignore[assignment]  # one connection, so the rows stay in this transaction
        jobs, _ = await q.list_jobs(db.JobFilter(site_url=SITE, sort="match"), 1, 50, user_id=USER_ID,
                                    skills=skills, titles=roles.patterns())
        return [(j.position, j.match_count) for j in jobs]
    finally:
        await tx.rollback()
        await conn.close()


def _ranked(dsn: str, skills: list[str], roles: RoleKeywords) -> list[tuple[str, int | None]]:
    # Playwright's sync API owns this thread's event loop; run on another.
    with ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, _rank(dsn, skills, roles)).result()


def test_a_devops_platform_seeker_gets_devops_and_platform_titles_first(e2e_dsn):
    roles = role_keywords("DevOps / Platform Engineer", ["Linux system administrator", "IT help desk"])
    ranked = _ranked(e2e_dsn, DEVOPS_SKILLS, roles)
    assert [p for p, _ in ranked] == [
        "Senior DevOps Engineer",       # headline role, 3 of 4 tags
        "Platform Engineer",            # headline role, 1 of 1
        "Linux Systems Administrator",  # recent role, 2 of 2
        "Site Reliability Engineer",    # no role named, 3 of 3
        "Technical Writer",             # 1 of 1, newest
        "Product Manager",              # 1 of 1
        "Senior SRE",                   # 1 of 2
        "DevOps Engineer",              # expired: last, despite 3 of 3 and the headline role
    ]
    # The "N of M skills" counts still come along.
    assert dict(ranked)["Senior DevOps Engineer"] == 3 and dict(ranked)["Technical Writer"] == 1


def test_an_sre_seeker_gets_every_sre_title_before_any_devops_one(e2e_dsn):
    roles = role_keywords("", ["Senior Site Reliability Engineer", "Systems Engineer"])
    ranked = [p for p, _ in _ranked(e2e_dsn, SRE_SKILLS, roles)]
    assert ranked[:3] == ["Site Reliability Engineer", "Senior SRE", "Linux Systems Administrator"]
    assert ranked[-1] == "DevOps Engineer"


def test_a_resume_naming_no_role_keeps_the_overlap_order_with_expired_last(e2e_dsn):
    ranked = [p for p, _ in _ranked(e2e_dsn, DEVOPS_SKILLS, RoleKeywords())]
    assert ranked[:2] == ["Site Reliability Engineer", "Senior DevOps Engineer"]
    assert ranked[-1] == "DevOps Engineer"
