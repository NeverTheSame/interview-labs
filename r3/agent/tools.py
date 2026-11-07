"""The advisor's read-only tools over the jobs database and the run
service. Every tool is bounded in time, never writes, and answers with
compact plain text (at most ~4000 characters); a failing query becomes a
short "unavailable" line the model can relay, never an exception.

:class:`AdvisorDeps` lives here because the tools are what read it; the
advisor re-exports it."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, TypeVar

from pydantic_ai import RunContext

from r3 import db
from r3.db import Database, Facets, JobFilter
from r3.db.users import User
from r3.job_summary import extract_overview
from r3.runs.models import StatsSnapshot, WorkflowRun
from r3.runs.service import RunService
from r3.runs.stats import build_stats_snapshot, compute_stats_snapshot, job_queries, run_stats_or_none
from r3.settings import ControllerSettings

from .contracts import Action, JobCard, RouteDecision
from .router import STATUS_WORDS, normalize_roles

log = logging.getLogger("r3.agent.tools")

TOOL_TIMEOUT_S = 10.0
MAX_TOOL_CHARS = 4000
DESCRIPTION_CHARS = 1500
RESUME_CHARS = 3000  # the resume text my_resume hands the model; the whole answer stays under MAX_TOOL_CHARS

T = TypeVar("T")


@dataclass
class AdvisorDeps:
    """What one advisor turn can see: the database and run service (read
    only), the settings, the router's decision, the action the panel shows and
    the signed-in account (None: the local operator, as with sign-in off)."""

    database: Database
    runs: RunService
    settings: ControllerSettings
    decision: RouteDecision
    action: Action
    include_context: bool = True
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    facets: Facets | None = None  # fetched for the router; reused by search_jobs
    portal_summary: str | None = field(default=None, repr=False)  # filled once per turn by the instructions
    user: User | None = None
    resume_note: str | None = field(default=None, repr=False)  # filled once per turn by the instructions

    @property
    def user_id(self) -> int | None:
        """The account whose job statuses and runs the tools read; None (the
        shared view) for the local operator."""
        return scoped_user_id(self.user)

    @property
    def is_admin(self) -> bool:
        return self.user is None or self.user.is_admin

    @property
    def sees_postings(self) -> bool:
        """Whether this account reads job postings and their links (R3_JOB_POSTINGS)."""
        return self.settings.postings_for(is_admin=self.is_admin)


def scoped_user_id(user: User | None) -> int | None:
    """``user_id`` for the per-user queries: the account's id, or None for
    the local operator (id 0, sign-in off), which keeps the shared view."""
    return user.id if user is not None and user.id > 0 else None


async def stats_snapshot(database: Database, runs: RunService | None, days: int, user_id: int | None) -> StatsSnapshot:
    """:func:`compute_stats_snapshot` with the job counts of one account
    (``user_id``); run statistics only when ``runs`` is given."""
    if user_id is None:
        return await compute_stats_snapshot(database, runs, days=days)
    (stats, facets), run_stats = await asyncio.gather(asyncio.gather(*job_queries(database, user_id)),
                                                      run_stats_or_none(runs, days))
    return build_stats_snapshot(stats, facets, days, run_stats)


class _Unavailable(Exception):
    """A tool's query failed or timed out."""


async def _bounded(awaitable: Awaitable[T], what: str) -> T:
    try:
        async with asyncio.timeout(TOOL_TIMEOUT_S):
            return await awaitable
    except Exception as exc:
        log.warning("Advisor tool query failed: %s (%s)", what, type(exc).__name__)
        raise _Unavailable(what) from exc


def _unavailable(what: str) -> str:
    return f"Unavailable: {what} did not answer. Tell the user this data could not be loaded right now."


def _clip(text: str, n: int = MAX_TOOL_CHARS) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _aware(t: datetime) -> datetime:
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def ago(t: datetime | None, now: datetime | None = None) -> str:
    """ "3h ago", "2d ago", "never"."""
    if t is None:
        return "never"
    seconds = ((now or datetime.now(timezone.utc)) - _aware(t)).total_seconds()
    if seconds < 90:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 48 * 3600:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "running"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def job_link(job: db.Job | JobCard) -> str:
    """ "[Position at Company](/jobs/<id>)": how the advisor names a job (OQ-7)."""
    name = job.position + (f" at {job.company}" if job.company else "")
    return f"[{name.replace('[', '(').replace(']', ')')}](/jobs/{job.id})"


def job_line(job: db.Job, now: datetime) -> str:
    """ "[position at company](/jobs/<id>) | location | salary | status | scraped <ago>"."""
    parts = [job_link(job).replace("|", "/"), job.location or "-", job.salary or "-", job.status or "-",
             f"scraped {ago(job.scraped_at, now)}"]
    return " | ".join(p.replace("|", "/").strip() for p in parts)


# --- tools ------------------------------------------------------------------------------


async def portal_overview(ctx: RunContext[AdvisorDeps]) -> str:
    """Headline numbers of the job search: jobs stored, not yet reviewed and
    applied, jobs added in the last 7 and 30 days, the job sites, and the
    top roles and locations."""
    d, uid = ctx.deps.database, ctx.deps.user_id
    if not ctx.deps.is_admin:
        return await _own_overview(ctx)
    try:
        stats, facets = await _bounded(asyncio.gather(d.overview(user_id=uid), d.job_facets(user_id=uid)),
                                       "the portal overview")
    except _Unavailable:
        return _unavailable("The portal overview")
    lines = [
        f"Jobs stored: {stats.total_jobs:,}; unreviewed (new): {stats.new_jobs:,}; applied: {stats.applied_jobs:,}.",
        f"Added in the last 7 days: {stats.jobs_last7:,}; last 30 days: {stats.jobs_last30:,}.",
        f"Job sites: {stats.total_sites}; newest job added {ago(stats.last_scrape_at, ctx.deps.now)}.",
        "Statuses: " + ", ".join(f"{v.value} {v.count:,}" for v in facets.statuses),
        "Top roles: " + ", ".join(f"{v.value} {v.count:,}" for v in facets.role_categories[:8]),
        "Top locations: " + ", ".join(f"{v.value} {v.count:,}" for v in facets.location_kinds[:8]),
    ]
    return _clip("\n".join(lines))


async def _own_overview(ctx: RunContext[AdvisorDeps], what: str = "Your job counts") -> str:
    """A regular account's numbers: its own triage counts and the size of
    the shared job corpus, never every account's runs."""
    d, uid = ctx.deps.database, ctx.deps.user_id
    try:
        stats, facets = await _bounded(asyncio.gather(d.overview(user_id=uid), d.job_facets(user_id=uid)),
                                       what.lower())
    except _Unavailable:
        return _unavailable(what)
    statuses = {v.value: v.count for v in facets.statuses}
    lines = [
        f"Jobs in R3 (one collection shared by every account): {stats.total_jobs:,}; added in the last 7 days: "
        f"{stats.jobs_last7:,}.",
        f"This user's own triage: {stats.new_jobs:,} not reviewed yet (new), {stats.applied_jobs:,} marked Applied, "
        f"{statuses.get('skipped', 0):,} marked Not for me.",
        "Top roles: " + ", ".join(f"{v.value} {v.count:,}" for v in facets.role_categories[:8]),
        "Top locations: " + ", ".join(f"{v.value} {v.count:,}" for v in facets.location_kinds[:8]),
    ]
    return _clip("\n".join(lines))


async def search_jobs(ctx: RunContext[AdvisorDeps], query: str = "", role: str = "", remote: bool = False,
                      status: str = "new", days: int = 0, limit: int = 10) -> str:
    """Search the stored jobs, newest first.

    Args:
        query: Words to find in the position, company or description (empty for any).
        role: A role category such as SRE, DevOps, Data, AI or ML (empty for any).
        remote: Only remote jobs.
        status: new (unreviewed), applied, skipped, expired, or "all".
        days: Only jobs added in the last N days (0 for any age).
        limit: How many jobs to return, 1 to 25.
    """
    deps = ctx.deps
    status = status.strip().lower().replace("-", "_").replace(" ", "_")
    status = {"unreviewed": "new", "any": "all", "": "all"}.get(status, status)
    f = JobFilter(search=query.strip()[:200], status="" if status == "all" else status, remote=bool(remote),
                  days=max(0, min(int(days or 0), 365)), sort="newest")
    if role.strip():
        known, unknown = normalize_roles([role], deps.facets)
        if known:
            f.role_category, f.role_categories = known[0], known if len(known) > 1 else []
        elif unknown and not f.search:
            f.search = unknown[0]
    limit = max(1, min(int(limit or 10), 25))
    try:
        jobs, page = await _bounded(deps.database.list_jobs(f, 1, limit, user_id=deps.user_id), "the job search")
    except _Unavailable:
        return _unavailable("The job search")
    if not jobs:
        return f"No jobs match (filter {f.query_string()})."
    head = f"{page.total:,} matching jobs (filter {f.query_string()}), showing {len(jobs)}:"
    return _clip("\n".join([head, *(job_line(j, deps.now) for j in jobs)]))


async def job_details(ctx: RunContext[AdvisorDeps], job_id: int) -> str:
    """One job in detail: position, company, location, salary, status,
    whether it is live, tech tags and what the job is about.

    Args:
        job_id: The job's id (the number in its /jobs/<id> link).
    """
    try:
        job = await _bounded(ctx.deps.database.get_job(int(job_id), user_id=ctx.deps.user_id), "the job lookup")
    except _Unavailable:
        return _unavailable("The job lookup")
    if job is None:
        return f"No job #{job_id}."
    now = ctx.deps.now
    posted = f"; posted {job.posted_date}" if job.posted_date else ""
    lines = [
        job_link(job),
        f"Location: {job.location or '-'} ({job.location_kind or 'unknown kind'})",
        f"Salary: {job.salary or 'not stated'}",
        f"Status: {job.status}" + (f", applied {ago(job.applied_at, now)}" if job.applied_at else ""),
        f"Role category: {job.role_category or '-'}; employment: {job.employment or '-'}",
    ]
    url = f"URL: {job.job_url}" + (f" (apply: {job.apply_url})" if job.apply_url and job.apply_url != job.job_url
                                   else "")
    if ctx.deps.is_admin:
        lines += [f"Site: {job.site_name}; scraped {ago(job.scraped_at, now)}{posted}", url]
    else:
        # A regular account never gets the source board. With
        # R3_JOB_POSTINGS=teaser it gets no posting or link either (OQ-6), so
        # neither does the model talking to it.
        lines.append(f"Live: {'yes' if job.live else 'no, the posting is gone'}; found {ago(job.scraped_at, now)}"
                     f"{posted}")
        if ctx.deps.sees_postings:
            lines.append(url)
    lines.append("Tech tags: " + (", ".join(job.tech_tags) if job.tech_tags else "-"))
    if job.notes:
        lines.append(f"The user's notes: {job.notes[:300]}")
    if not ctx.deps.sees_postings:
        lines.append(f"Overview: {await _overview(ctx.deps.database, job)}")
        return _clip("\n".join(lines))
    description = " ".join(job.description.split())
    if description:
        cut = description[:DESCRIPTION_CHARS]
        lines.append("Description: " + cut + ("…" if len(description) > DESCRIPTION_CHARS else ""))
    return _clip("\n".join(lines))


async def _overview(database: Database, job: db.Job) -> str:
    """The job page's "About this role" text: the cached AI overview, else
    the extract. Never a model call from inside a tool."""
    try:
        cached = await _bounded(database.get_job_summary(job.id), "the job overview")
    except _Unavailable:
        cached = None
    return cached.summary if cached is not None else extract_overview(job).text


def run_summary(run: WorkflowRun) -> str:
    """One line for a run's outcome, from its typed result."""
    result: dict[str, Any] = run.result or {}
    if run.error:
        return f"error: {' '.join(run.error.split())[:160]}"
    if not result:
        progress = run.progress
        return progress.message[:160] if progress and progress.message else ("in progress" if run.open else "no result")
    if result.get("kind", run.kind) == "stats":
        return f"{result.get('total_jobs', 0)} jobs, {result.get('new_jobs', 0)} new, {result.get('applied_jobs', 0)} applied"
    return str(result.get("message", ""))[:160] or "done"


async def recent_runs(ctx: RunContext[AdvisorDeps], limit: int = 5) -> str:
    """The most recent workflow runs (statistics snapshots), newest first,
    with their status, timing and outcome.

    Args:
        limit: How many runs, 1 to 20.
    """
    limit = max(1, min(int(limit or 5), 20))
    try:
        user_id = None if ctx.deps.is_admin else ctx.deps.user_id  # admins see every account's runs
        runs = await _bounded(ctx.deps.runs.list(limit, user_id=user_id) if user_id is not None
                              else ctx.deps.runs.list(limit), "the run service")
    except _Unavailable:
        return _unavailable("The run service")
    if not runs:
        return "No workflow runs recorded yet."
    now = ctx.deps.now
    # Run ids are internal; a regular account's lines describe the run by kind.
    lines = [f"{'' if not ctx.deps.is_admin else r.id + ' | '}{r.kind} | {r.status} | started {ago(r.started_at, now)} | "
             f"{_duration(r.duration_s)} | "
             f"{run_summary(r)}" + (f" | note: {r.note[:80]}" if r.note else "") for r in runs]
    return _clip("\n".join(lines))


async def pipeline_stats(ctx: RunContext[AdvisorDeps], days: int = 30) -> str:
    """Statistics over the last N days: job counts by status, top roles and
    locations, and how the workflow runs went.

    Args:
        days: The window in days, 1 to 365.
    """
    days = max(1, min(int(days or 30), 365))
    deps = ctx.deps
    if not deps.is_admin:
        return await _own_overview(ctx, "Statistics")
    try:
        snap = await _bounded(stats_snapshot(deps.database, deps.runs, days, deps.user_id), "the statistics queries")
    except _Unavailable:
        return _unavailable("Statistics")
    lines = [
        f"Window: last {snap.window_days} days (captured {ago(snap.captured_at, deps.now)}).",
        f"Jobs: {snap.total_jobs:,} stored, {snap.new_jobs:,} new, {snap.applied_jobs:,} applied; "
        f"{snap.jobs_last7:,} added in 7 days, {snap.jobs_last30:,} in 30. Newest job added "
        f"{ago(snap.last_scrape_at, deps.now)}.",
        "Statuses: " + ", ".join(f"{STATUS_WORDS.get(k, k)} {v:,}" for k, v in snap.statuses.items()),
        "Top roles: " + ", ".join(f"{c.value} {c.count:,}" for c in snap.top_roles),
        "Top locations: " + ", ".join(f"{c.value} {c.count:,}" for c in snap.top_locations),
    ]
    if snap.runs is not None:
        r = snap.runs
        kinds = "; ".join(f"{kind}: " + ", ".join(f"{n} {status}" for status, n in by.items())
                          for kind, by in r.by_kind.items())
        lines.append(f"Runs in the window: {r.total} total, {r.running} running, {r.failed} failed"
                     + (f" ({kinds})" if kinds else "") + ".")
    else:
        lines.append("Run statistics are unavailable.")
    return _clip("\n".join(lines))


async def my_resume(ctx: RunContext[AdvisorDeps]) -> str:
    """The user's resume as R3 has it: whether one exists; if so the primary
    resume's title, source, age, length and text. Call it before advice on positioning, fit or which jobs suit them."""
    deps = ctx.deps
    account = deps.user.id if deps.user is not None else 0
    try:
        resume = await _bounded(deps.database.primary_resume(account), "the resume lookup")
    except _Unavailable:
        return _unavailable("The resume lookup")
    if resume is None:
        return ("The user has no resume in R3 yet. The fastest way: the \"Upload resume\" button under this chat "
                "(or drop the file on the chat) takes a PDF, DOCX, TXT or MD file, and R3 reads it and asks here for "
                "anything missing. They can also upload or paste it on the [Resume](/resume) page, or [build one by "
                "answering a few questions](/resume/build).")
    text = resume.content.strip()
    lines = [
        f"Primary resume: {resume.title or 'untitled'}; source: {resume.source}; updated "
        f"{ago(resume.updated_at, deps.now)}; {len(text):,} characters.",
        "Resume text" + (" (clipped)" if len(text) > RESUME_CHARS else "") + ":",
        _clip(text, RESUME_CHARS),
    ]
    return _clip("\n".join(lines))


ADVISOR_TOOLS = (portal_overview, search_jobs, job_details, my_resume, recent_runs, pipeline_stats)
