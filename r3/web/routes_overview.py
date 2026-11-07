"""The job teaser's "About this role" overview (OQ-6): ``POST
/jobs/{id}/overview`` answers with the cached overview or asks the
summarizer (:mod:`r3.job_summary`, ``app.state.job_summarizer``) for one and
caches it in ``r3_job_summaries`` when a model wrote it.

A regular account never sees the posting: the job page shows this overview
instead of the description. With htmx the route returns the overview
fragment; a plain form post is redirected back to the job page.

Model calls are spent carefully: clicks on one job while its overview is
being written share that one call, and after a fallback (no key, a failure,
a timeout) the job answers with the extract for :data:`FALLBACK_TTL_S`
instead of asking the model again. That same record is the only way the
page shows the extract without a click; nothing in the URL can.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from r3.events import record as record_event
from r3.db.jobs import Job
from r3.job_summary import Overview, extract_overview

from .auth import User, current_user
from .inputs import DbId

router = APIRouter(tags=["overview"])
log = logging.getLogger(__name__)

# How long a job keeps answering with the extract after a fallback.
FALLBACK_TTL_S = 600.0


def init_state(app: FastAPI) -> None:
    """The per-app bookkeeping: overviews being written (job id -> task) and
    recent fallbacks (job id -> monotonic time)."""
    app.state.overview_inflight = {}
    app.state.overview_fallbacks = {}


async def cancel_overviews(app: FastAPI) -> None:
    """Stop overview calls still running at shutdown."""
    tasks = list(getattr(app.state, "overview_inflight", {}).values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _recent_fallback(app: FastAPI, job_id: int) -> bool:
    fallbacks: dict[int, float] = app.state.overview_fallbacks
    at = fallbacks.get(job_id)
    if at is None:
        return False
    if time.monotonic() - at < FALLBACK_TTL_S:
        return True
    del fallbacks[job_id]
    return False


def _note_fallback(app: FastAPI, job_id: int) -> None:
    fallbacks: dict[int, float] = app.state.overview_fallbacks
    now = time.monotonic()
    for stale in [j for j, at in fallbacks.items() if now - at >= FALLBACK_TTL_S]:
        del fallbacks[stale]
    fallbacks[job_id] = now


async def cached_overview(request: Request, job_id: int) -> Overview | None:
    """The stored model-written overview, or None (none yet, or the cache is
    unreadable: the page then offers the button instead of failing)."""
    try:
        row = await request.app.state.db.get_job_summary(job_id)
    except Exception as e:
        log.warning("job overview cache unavailable: %s: %s", type(e).__name__, e)
        return None
    return None if row is None else Overview(text=row.summary, engine="llm", model=row.model)


async def page_overview(request: Request, job: Job) -> dict[str, Any]:
    """Template context for a seeker's job page: the cached overview, the
    extract while a recent fallback stands, else nothing (the page shows the
    button). Never calls the model."""
    overview = await cached_overview(request, job.id)
    if overview is None and _recent_fallback(request.app, job.id):
        overview = extract_overview(job)
    return {"overview": overview}


async def _write(app: FastAPI, job: Job) -> Overview:
    overview = await app.state.job_summarizer.overview(job)
    if overview.engine != "llm":
        _note_fallback(app, job.id)
        return overview
    try:
        await app.state.db.save_job_summary(job.id, overview.text, overview.model)
    except Exception as e:  # the seeker still gets the overview; the next click asks again
        log.warning("job overview not cached: %s: %s", type(e).__name__, e)
    return overview


async def overview_for(request: Request, job: Job) -> Overview:
    """The cached overview, the extract during a fallback window, or a new
    one; concurrent requests for one job share a single summarizer call."""
    app = request.app
    cached = await cached_overview(request, job.id)
    if cached is not None:
        return cached
    if _recent_fallback(app, job.id):
        return extract_overview(job)
    inflight: dict[int, asyncio.Task[Overview]] = app.state.overview_inflight
    task = inflight.get(job.id)
    if task is None:
        task = asyncio.ensure_future(_write(app, job))
        inflight[job.id] = task
        task.add_done_callback(lambda _: inflight.pop(job.id, None))
    # Shielded: a seeker who navigates away does not cancel the call others wait on.
    return await asyncio.shield(task)


@router.post("/jobs/{job_id}/overview")
async def job_overview(request: Request, job_id: DbId, user: User = Depends(current_user)) -> Response:
    job = await request.app.state.db.get_job(job_id, user_id=user.id)
    if job is None:
        raise HTTPException(404, "job not found")
    cached = await cached_overview(request, job.id) is not None
    overview = await overview_for(request, job)
    record_event("job.overview", f"R3 wrote an overview of “{job.position}” ({overview.engine})", actor="ai",
                 user_id=user.id, subject=f"job:{job.id}",
                 data={"engine": overview.engine, "model": overview.model, "cached": cached,
                       "chars": len(overview.text)})
    if request.headers.get("HX-Request") == "true":
        tmpl = request.app.state.templates.get_template("_job_overview.html")
        return HTMLResponse(tmpl.render(job=job, overview=overview, swap=True))
    return RedirectResponse(f"{request.scope['root_path']}/jobs/{job.id}#about", status_code=303)
