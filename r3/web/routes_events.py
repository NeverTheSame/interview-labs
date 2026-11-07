"""R3's audit trail on the web.

``POST /api/events`` takes the clicks, form submits, page views and choices a
person makes in the browser, sent by static/track.js, and records them as
``ui.*`` audit events (r3/events.py) for the signed-in account. The page sends
what was clicked (its visible label, the element, the page and, for links,
where it leads), never what was typed. Batches are capped at 50 and each
account at 600 events per 10 minutes, so a broken or hostile page cannot
flood the table.

The trail is read back in two places: a job page's "What happened" (that
account's decisions about that job) and the admin Events page (everything,
filterable down to one trace).
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from r3.events import current_trace, default_log

from .auth import User, current_user, require_admin

log = logging.getLogger("r3.web.events")

router = APIRouter()

WINDOW_S = 600
WINDOW_MAX = 600
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_JOB_PAGE = re.compile(r"^/jobs/(\d+)(?:[/?#]|$)")
_RESUME_PAGE = re.compile(r"^/resume/(\d+)(?:[/?#]|$)")
_VERBS = {"click": "Clicked", "submit": "Submitted", "change": "Changed", "view": "Viewed"}


def _short(limit: int):
    def clip(v: str) -> str:
        v = " ".join(str(v or "").split())
        return v if len(v) <= limit else v[: limit - 1] + "…"
    return clip


class UiEvent(BaseModel):
    type: Literal["click", "submit", "change", "view"]
    label: str = ""
    target: str = ""
    path: str = ""
    href: str = ""
    trace: str = ""

    _label = field_validator("label", mode="before")(lambda cls, v: _short(120)(v))
    _target = field_validator("target", mode="before")(lambda cls, v: _short(120)(v))
    _path = field_validator("path", "href", mode="before")(lambda cls, v: _short(300)(v))


class UiBatch(BaseModel):
    events: list[UiEvent] = Field(max_length=50)


def _subject(path: str) -> str:
    if m := _JOB_PAGE.match(path):
        return f"job:{m.group(1)}"
    if m := _RESUME_PAGE.match(path):
        return f"resume:{m.group(1)}"
    return ""


def _allowed(request: Request, key: str, n: int) -> int:
    """How many of ``n`` events fit this account's window."""
    windows: dict[str, deque[float]] = request.app.state.__dict__.setdefault("ui_event_windows", {})
    now = time.monotonic()
    w = windows.setdefault(key, deque())
    while w and now - w[0] > WINDOW_S:
        w.popleft()
    room = max(0, WINDOW_MAX - len(w))
    for _ in range(min(n, room)):
        w.append(now)
    return min(n, room)


@router.post("/api/events", status_code=204)
async def ui_events(batch: UiBatch, request: Request, user: User = Depends(current_user)) -> Response:
    ctx = current_trace()
    keep = _allowed(request, f"user:{user.id}", len(batch.events))
    for ev in batch.events[:keep]:
        label = ev.label or ev.target or ev.type
        default_log.record(
            f"ui.{ev.type}", f"{_VERBS[ev.type]} “{label}” on {ev.path or '?'}", actor="user", user_id=user.id,
            trace_id=ev.trace if _HEX32.match(ev.trace) else (ctx.trace_id if ctx else None),
            subject=_subject(ev.path),
            data={"label": ev.label, "target": ev.target, "path": ev.path, "href": ev.href})
    return Response(status_code=204)


# --- reading the trail ---------------------------------------------------------------

# The kinds that tell a job's story; clicks, page views and raw requests are
# in the admin trail but would drown it.
_STORY_PREFIXES = ("job.", "resume.", "run.", "chat.")
_WHO = {"user": "You", "ai": "R3's AI", "system": "R3"}


async def job_history(request: Request, user_id: int, job_id: int) -> list[dict[str, Any]]:
    """What happened to one job for this account, newest first; [] when the
    trail cannot be read (the page still works)."""
    db = request.app.state.db
    if not hasattr(db, "list_events"):
        return []
    try:
        # Events reach the database in one-second batches; write what is
        # buffered first, so a change made just before this page shows on it.
        await default_log.flush()
        rows = await db.list_events(user_id=user_id, subject=f"job:{job_id}", limit=50)
    except Exception as e:
        log.warning("job history unavailable: %s: %s", type(e).__name__, e)
        return []
    return [{"at": r.at, "who": _WHO.get(r.actor, r.actor), "summary": r.summary, "kind": r.kind}
            for r in rows if r.kind.startswith(_STORY_PREFIXES)]


@router.get("/events", response_class=HTMLResponse, dependencies=[Depends(require_admin)])
async def events_page(request: Request, user: str = "", kind: str = "", subject: str = "", trace: str = "",
                      limit: int = 300) -> HTMLResponse:
    """Every recorded action and decision, newest first, filtered by account,
    kind (a prefix such as ``ai.`` or ``job.status``), subject or trace."""
    db = request.app.state.db
    uid = int(user) if user.strip().isdigit() else None
    clean_trace = trace.strip().lower() if _HEX32.match(trace.strip().lower()) else ""
    rows, error = [], ""
    if hasattr(db, "list_events"):
        try:
            rows = await db.list_events(user_id=uid, subject=subject.strip()[:80], trace_id=clean_trace,
                                        kind_prefix=kind.strip()[:60], limit=max(1, min(limit, 1000)))
        except Exception as e:
            error = f"The trail could not be read ({type(e).__name__})."
    else:
        error = "This portal's database has no audit trail."
    view = [{**r.model_dump(), "data_json": json.dumps(r.data, indent=2, ensure_ascii=False, default=str)}
            for r in rows]
    tmpl = request.app.state.templates.get_template("events.html")
    return HTMLResponse(tmpl.render(title="Events", nav="events", query={}, rows=view, error=error,
                                    filters={"user": user, "kind": kind, "subject": subject, "trace": clean_trace}))
