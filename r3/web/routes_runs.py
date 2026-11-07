"""Runs API: start, list and inspect workflow runs, statistics, the panel's
state (pipeline facts, suggestions, recent runs) and a JSON job list for
pickers. Owned by the runs side. Handlers read ``db``, ``runs`` and
``settings`` from ``request.app.state``."""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from starlette.datastructures import QueryParams

from r3 import db
from r3.agent.contracts import JobCard
from r3.panel import panel_state
from r3.runs.models import WORKFLOW_KINDS, PanelState, RunRequest, StatsSnapshot, WorkflowRun
from r3.runs.service import RunService, RunsUnavailable
from r3.runs.stats import compute_stats_snapshot, for_account, hide_run_internals

from .auth import LOCAL_USER, User, current_user
from .inputs import MAX_DAYS, MAX_PAGE, MAX_SALARY, clean_query

log = logging.getLogger("r3.web.runs")

router = APIRouter(prefix="/api", tags=["runs"])

JOBS_PAGE_DEFAULT = 20
JOBS_PAGE_MAX = 50


class CancelResult(BaseModel):
    cancelled: bool


class JobList(BaseModel):
    jobs: list[JobCard]
    total: int
    filter_query: str
    page: int = 1
    pages: int = 0


def _runs(request: Request) -> RunService:
    return request.app.state.runs


def _atoi(raw: str | None) -> int:
    """A non-negative int; anything else is 0 (as on the /jobs page)."""
    try:
        n = int((raw or "").strip())
    except ValueError:
        return 0
    return n if n >= 0 else 0


def parse_job_query(q: QueryParams) -> tuple[db.JobFilter, int, int]:
    """The /jobs page's query string as a filter, page and page size, with
    the same semantics: a repeated ``role`` means any of them, flags are
    ``=1``, numbers that do not parse are ignored. The page size is ``size``
    (or ``limit``, as the chat's route parameters call it), at most 50."""
    roles = q.getlist("role")
    f = db.JobFilter(
        search=q.get("q", ""),
        status=q.get("status", ""),
        role_category=roles[0] if roles else "",
        location_kind=q.get("location", ""),
        site_url=q.get("site", ""),
        tech_tag=q.get("tech", ""),
        employment=q.get("employment", ""),
        min_salary=min(_atoi(q.get("min_salary")), MAX_SALARY),
        days=min(_atoi(q.get("days")), MAX_DAYS),
        sort=q.get("sort", ""),
        employment_family=q.get("employment_family", ""),
        remote=q.get("remote") == "1",
        has_salary=q.get("has_salary") == "1",
    )
    if len(roles) > 1:
        f.role_categories = roles
    page = min(_atoi(q.get("page")) or 1, MAX_PAGE)
    size = min(_atoi(q.get("size")) or _atoi(q.get("limit")) or JOBS_PAGE_DEFAULT, JOBS_PAGE_MAX)
    return f, page, size


def run_owner(user: User) -> int | None:
    """The ``user_id`` a run started by ``user`` belongs to: None with
    sign-in off (the local operator), the account's id otherwise."""
    return None if user.id == LOCAL_USER.id else user.id


def _postings(request: Request, user: User) -> bool:
    """Whether this account sees job postings and their links (R3_JOB_POSTINGS)."""
    settings = getattr(request.app.state, "settings", None)
    return settings is None or settings.postings_for(is_admin=user.is_admin)


async def _own_run(request: Request, user: User, run_id: str) -> WorkflowRun:
    """The run, when ``user`` may see it: admins see every run, a regular
    account only its own. Anything else is "not found", never "forbidden",
    so run ids of other accounts are not confirmed."""
    run = await _runs(request).get(run_id)
    if run is None or (not user.is_admin and run.user_id != user.id):
        raise HTTPException(404, "run not found")
    return run


@router.post("/runs", status_code=202, response_model=WorkflowRun)
async def start_run(request: Request, body: RunRequest, user: User = Depends(current_user)) -> WorkflowRun:
    # Who the run belongs to comes from the session, whatever the client sent.
    # With sign-in off there are no accounts and the client's label stays.
    owner = run_owner(user)
    body = body.model_copy(update={"user_id": owner} if owner is None
                           else {"user_id": owner, "requested_by": user.account_name[:80]})
    try:
        run = await _runs(request).start(body)
    except RunsUnavailable as e:
        raise HTTPException(503, str(e) or "Runs are unavailable") from None
    return run if user.is_admin else hide_run_internals(run)


@router.get("/runs", response_model=list[WorkflowRun])
async def list_runs(request: Request, limit: int = Query(20, ge=1, le=100), kind: str = "",
                    user: User = Depends(current_user)) -> list[WorkflowRun]:
    if kind and kind not in WORKFLOW_KINDS:
        raise HTTPException(422, f"kind must be one of {', '.join(WORKFLOW_KINDS)}")
    if user.is_admin:
        return await _runs(request).list(limit, kind or None)  # type: ignore[arg-type]
    runs = await _runs(request).list(limit, kind or None, user_id=user.id)  # type: ignore[arg-type]
    return [hide_run_internals(r) for r in runs]


@router.get("/runs/{run_id}", response_model=WorkflowRun)
async def get_run(request: Request, run_id: str, user: User = Depends(current_user)) -> WorkflowRun:
    run = await _own_run(request, user, run_id)
    return run if user.is_admin else hide_run_internals(run)


@router.post("/runs/{run_id}/cancel", response_model=CancelResult)
async def cancel_run(request: Request, run_id: str, user: User = Depends(current_user)) -> CancelResult:
    if not user.is_admin:
        await _own_run(request, user, run_id)
    return CancelResult(cancelled=await _runs(request).cancel(run_id))


@router.get("/stats", response_model=StatsSnapshot)
async def stats(request: Request, days: int = Query(30, ge=1, le=365),
                user: User = Depends(current_user)) -> StatsSnapshot:
    # Admins see the whole portal; a regular account its own jobs and runs.
    try:
        if user.is_admin:
            return await compute_stats_snapshot(request.app.state.db, _runs(request), days)
        snapshot = await compute_stats_snapshot(request.app.state.db, _runs(request), days, user_id=user.id)
    except Exception:
        log.exception("statistics failed")
        raise HTTPException(503, "Statistics are unavailable: the database did not answer") from None
    return for_account(snapshot)


@router.get("/panel", response_model=PanelState)
async def panel(request: Request, user: User = Depends(current_user)) -> PanelState:
    if user.is_admin:
        return await panel_state(request.app.state.db, _runs(request), request.app.state.settings)
    return await panel_state(request.app.state.db, _runs(request), request.app.state.settings,
                             user_id=user.id, operator=False)


@router.get("/jobs", response_model=JobList)
async def jobs(request: Request, user: User = Depends(current_user)) -> Any:
    # Status and notes (and a ?status= filter) are the caller's own triage.
    f, page, size = parse_job_query(clean_query(request.query_params))
    try:
        rows, pg = await request.app.state.db.list_jobs(f, page, size, user_id=user.id)
    except asyncpg.DataError as e:
        raise HTTPException(400, f"Bad filter value: {e}") from None
    except Exception:
        log.exception("job list failed")
        raise HTTPException(503, "Jobs are unavailable: the database did not answer") from None
    return JobList(jobs=[JobCard.from_job(j, full=_postings(request, user)) for j in rows], total=pg.total, filter_query=f.query_string(),
                   page=pg.number, pages=pg.pages)
