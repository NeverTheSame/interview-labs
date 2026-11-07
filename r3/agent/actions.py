"""From a route to what the panel shows. :func:`plan_action` turns a
:class:`RouteDecision` into an :class:`Action`: it lists ``jobs``, snapshots
``stats`` (recorded as a run when a runner is available), and reports plainly
when the database does not answer.

Every database call is bounded; a database that does not answer becomes an
:class:`UnavailableAction` with a safe reason, never an exception."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, TypeVar

from r3.db import Database, Facets, JobFilter
from r3.db.users import User
from r3.runs.models import RunRequest, StatsParams, WorkflowRun
from r3.runs.service import RunService, RunsUnavailable
from r3.settings import ControllerSettings

from .contracts import (
    Action,
    AdviceAction,
    JobCard,
    JobsAction,
    RouteDecision,
    RouteParams,
    StatsAction,
    UnavailableAction,
)
from .tools import scoped_user_id, stats_snapshot
from .router import STATUS_WORDS, employment_family, normalize_roles, normalize_tech

log = logging.getLogger("r3.agent.actions")

DB_TIMEOUT_S = 10.0
STATS_TIMEOUT_S = 15.0  # the snapshot also waits (at most 8s) for run statistics
RUN_START_TIMEOUT_S = 10.0
FACETS_TIMEOUT_S = 5.0
DEFAULT_LIST = 10

T = TypeVar("T")

DB_DOWN = "The database did not answer"
DB_HINT = "Try again in a moment; the chat keeps working without it."


class DatabaseUnavailable(Exception):
    """A bounded database call failed or timed out."""


async def _db(awaitable: Awaitable[T], timeout: float = DB_TIMEOUT_S) -> T:
    try:
        async with asyncio.timeout(timeout):
            return await awaitable
    except Exception as exc:
        log.warning("Database call failed while planning an action (%s)", type(exc).__name__)
        raise DatabaseUnavailable from exc


async def fetch_facets(database: Database, timeout: float = FACETS_TIMEOUT_S, *,
                       user_id: int | None = None) -> Facets | None:
    """The filter vocabulary (role categories, tech tags, statuses), or None."""
    try:
        async with asyncio.timeout(timeout):
            return await database.job_facets(user_id=user_id)
    except Exception as exc:
        log.warning("Job facets unavailable (%s)", type(exc).__name__)
        return None


async def plan_action(decision: RouteDecision, *, database: Database, runs: RunService, settings: ControllerSettings,
                      model_id: str, text: str, facets: Facets | None = None, user: User | None = None) -> Action:
    """The panel action for a route. ``facets`` (the database's filter
    vocabulary) are fetched when not given; they map roles and tech onto the
    database's own spellings. ``user`` is the account asking: job lists use
    its statuses and the stats run carries its id (None: the local operator)."""
    intent = decision.intent
    if intent == "advice":
        return AdviceAction()
    user_id = scoped_user_id(user)
    try:
        if facets is None:
            facets = await fetch_facets(database, user_id=user_id)
        if intent == "jobs":
            return await _jobs(decision.params, database, facets, user_id,
                               full=user is None or settings.postings_for(is_admin=user.is_admin))
        return await _stats(decision.params, database, runs, model_id, text, user)
    except DatabaseUnavailable:
        return UnavailableAction(intent=intent, reason=DB_DOWN, hint=DB_HINT)


# --- the jobs filter --------------------------------------------------------------------


def _location_kind(location: str, remote: bool | None, facets: Facets | None) -> str:
    """A location kind (``remote-us`` ...) for "US"/"Canada"/"worldwide"/"hybrid",
    when exactly one of the database's kinds fits."""
    loc = location.strip().lower()
    if not loc:
        return ""
    words = set(loc.replace("-", " ").replace(".", "").split())
    if "canada" in loc or words & {"ca", "canadian"}:
        want = "ca"
    elif words & {"us", "usa"} or "united states" in loc or "america" in loc:
        want = "us"
    elif words & {"worldwide", "anywhere", "global", "globally", "ww", "international"}:
        want = "ww"
    elif "hybrid" in loc:
        want = "hybrid"
    elif "onsite" in loc or "on site" in loc or "office" in loc:
        want = "onsite"
    else:
        return ""
    kinds = [v.value for v in facets.location_kinds if v.value] if facets else []
    if not kinds:
        return f"remote-{want}" if remote and want in ("us", "ca", "ww") else ""

    def fits(kind: str) -> bool:
        parts = kind.lower().split("-")
        if want in ("us", "ca", "ww"):
            return want in parts or (want == "ca" and "canada" in kind.lower()) or (want == "ww" and "worldwide" in kind.lower())
        return want in kind.lower() or (want == "onsite" and "office" in kind.lower())

    candidates = [k for k in kinds if fits(k)]
    if remote is True:
        candidates = [k for k in candidates if k.lower().startswith("remote")] or candidates
    elif remote is False:
        candidates = [k for k in candidates if not k.lower().startswith("remote")] or candidates
    return candidates[0] if len(candidates) == 1 else ""


def job_filter(params: RouteParams, facets: Facets | None, *, default_status: str = "new") -> JobFilter:
    """The /jobs filter a route describes. Roles map onto the database's
    categories (a role it does not know becomes the search text); a tech tag
    the database does not have is searched for as text instead."""
    status = params.status or default_status
    f = JobFilter(status="" if status == "all" else status, days=params.days or 0, min_salary=params.min_salary or 0)
    known, unknown = normalize_roles(params.roles, facets)
    if known:
        f.role_category = known[0]
        if len(known) > 1:
            f.role_categories = known
    search = params.search.strip()
    if params.tech:
        tag = normalize_tech(params.tech[0], facets)
        if facets and facets.tech_tags and all(v.value != tag for v in facets.tech_tags):
            search = search or params.tech[0].strip()
        else:
            f.tech_tag = tag
    if not search and unknown:
        search = unknown[0]
    f.search = search[:200]
    if params.remote:
        f.remote = True
    f.location_kind = _location_kind(params.location, params.remote, facets)
    if params.employment:
        f.employment_family = employment_family(params.employment)
    if f.min_salary > 0:
        f.sort = "salary"  # what the /jobs link does too
    return f


def _location_label(kind: str) -> str:
    names = {"us": "US", "ca": "Canada", "ww": "worldwide"}
    return " ".join(names.get(part, part) for part in kind.split("-"))


def jobs_title(f: JobFilter, total: int) -> str:
    """ "12 new remote SRE jobs", "3 applied jobs with python, from the last 7 days"."""
    words: list[str] = []
    if status := STATUS_WORDS.get(f.status, f.status):
        words.append(status)
    location = _location_label(f.location_kind) if f.location_kind else ""
    if f.remote and not location.startswith("remote"):
        words.append("remote")
    if location:
        words.append(location)
    if f.employment_family:
        words.append(f.employment_family)
    roles = f.role_categories or ([f.role_category] if f.role_category else [])
    if roles:
        words.append("/".join(roles))
    title = " ".join([f"{total:,}", *words, "job" if total == 1 else "jobs"])
    extras: list[str] = []
    if f.tech_tag:
        extras.append(f"with {f.tech_tag}")
    if f.search:
        extras.append(f'matching "{f.search}"')
    if f.min_salary:
        extras.append(f"paying ${f.min_salary // 1000}k+")
    if f.days:
        extras.append("from the last day" if f.days == 1 else f"from the last {f.days} days")
    return title + (" " + ", ".join(extras) if extras else "")


async def _jobs(params: RouteParams, database: Database, facets: Facets | None, user_id: int | None, *,
                full: bool) -> JobsAction:
    if params.job_ids:
        # Named jobs ("tell me about job 7") are shown as asked; there is no /jobs query for ids.
        ids = list(dict.fromkeys(params.job_ids))[:25]
        found = await _db(asyncio.gather(*(database.get_job(i, user_id=user_id) for i in ids)))
        jobs = [job for job in found if job is not None]
        missing = [f"#{i}" for i, job in zip(ids, found, strict=True) if job is None]
        title = ", ".join(f"#{j.id} {j.position}" for j in jobs[:3]) + ("…" if len(jobs) > 3 else "")
        if missing:
            title = (title + "; " if title else "") + "not found: " + ", ".join(missing)
        return JobsAction(title=title or "No jobs", jobs=[JobCard.from_job(j, full=full) for j in jobs],
                          total=len(jobs))
    f = job_filter(params, facets)
    jobs, page = await _db(database.list_jobs(f, 1, params.limit or DEFAULT_LIST, user_id=user_id))
    return JobsAction(title=jobs_title(f, page.total), filter_query=f.query_string(),
                      jobs=[JobCard.from_job(j, full=full) for j in jobs], total=page.total)


# --- runs ---------------------------------------------------------------------------------


def _request(params: StatsParams, model_id: str, text: str, user: User | None = None) -> RunRequest:
    """A chat run, owned by ``user`` (its id and account name) unless it is
    the local operator."""
    owner = {"user_id": user.id, "requested_by": user.account_name[:80]} if scoped_user_id(user) is not None else {}
    return RunRequest(params=params, source="chat", model=model_id[:160], note=" ".join(text.split())[:300], **owner)


async def _stats(params: RouteParams, database: Database, runs: RunService, model_id: str, text: str,
                 user: User | None = None) -> StatsAction:
    days = params.days or 30
    admin = user is None or user.is_admin  # a regular account never sees every account's run statistics
    snapshot = await _db(stats_snapshot(database, runs if admin else None, days, scoped_user_id(user)),
                         timeout=STATS_TIMEOUT_S)
    # Recording the snapshot as a stats run is best effort: the numbers above
    # stand on their own when no runner is available.
    run: WorkflowRun | None = None
    try:
        async with asyncio.timeout(RUN_START_TIMEOUT_S):
            run = await runs.start(_request(StatsParams(days=days), model_id, text, user))
    except RunsUnavailable:
        run = None
    except Exception as exc:
        log.info("Stats snapshot not recorded as a run (%s)", type(exc).__name__)
        run = None
    return StatsAction(snapshot=snapshot, run=run)
