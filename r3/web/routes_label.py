"""The labeling page (``/label``): one scraped posting at a time, the rubric
in :mod:`r3.labeling` as one-key answers, the labels in ``r3_job_labels``
(:mod:`r3.db.labels`), and ``/label/export`` as JSONL for the alignment eval.

The page is server-rendered; HTMX swaps the card (``/label/next``, the save
``POST``) and, out of band, the counter and the flash line. Session state
that must not reach the database (the jobs skipped this session, the undo
stack) lives in the browser and comes along as form fields: ``seen`` (ids
shown this session), ``since`` (the labeler's local midnight, for "labeled
today"), ``session_id`` and ``seconds_spent``. ``scraped_jobs`` is never
written here.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Mapping

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from r3 import db
from r3.web.inputs import MAX_DAYS, DbId
from r3.labeling import (
    DEFAULT_LABELER,
    RUBRIC,
    RUBRIC_VERSION,
    SKIPPED,
    clean_note,
    clean_seconds,
    clean_session_id,
    parse_answers,
    summarize,
)

from .auth import require_admin

log = logging.getLogger("r3.web.label")

# Operator console: every route here is for admins only.
router = APIRouter(tags=["label"], dependencies=[Depends(require_admin)])

SEEN_MAX = 2000  # ids a request may carry; the browser sends the most recent ones


def _db(request: Request) -> Any:
    return request.app.state.db


def _query(request: Request) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in request.query_params.multi_items():
        out.setdefault(k, []).append(v)
    return out


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def _page(request: Request, **ctx: Any) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template("label.html")
    return HTMLResponse(tmpl.render(title="Label", nav="label", query=_query(request), **ctx))


def _fragment(request: Request, **ctx: Any) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template("label_fragment.html")
    return HTMLResponse(tmpl.render(query=_query(request), **ctx))


def _atoi(s: str | None) -> int:
    try:
        n = int((s or "").strip())
    except ValueError:
        return 0
    return n if n >= 0 else 0


def label_filter(values: Mapping[str, Any]) -> db.JobFilter:
    """The slice being labeled, from the query string or the posted form:
    role category, location kind, employment, site and days."""
    return db.JobFilter(
        role_category=str(values.get("role") or ""),
        location_kind=str(values.get("location") or ""),
        employment=str(values.get("employment") or ""),
        site_url=str(values.get("site") or ""),
        days=min(_atoi(values.get("days")), MAX_DAYS),
    )


def parse_seen(raw: Any) -> list[int]:
    """``"7,9,12"`` -> ``[7, 9, 12]``; anything unparseable is dropped."""
    out: list[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if part.isdigit() and int(part) < 2**63:
            out.append(int(part))
    return list(dict.fromkeys(out))[-SEEN_MAX:]


def parse_since(raw: Any) -> datetime:
    """The start of the labeler's day (ISO 8601 from the browser), else UTC midnight."""
    text = str(raw or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            parsed = None
        if parsed is not None:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def checked_values(existing: db.Label | None) -> dict[str, set[str]]:
    """Which option of each question the card should show selected."""
    out: dict[str, set[str]] = {}
    if existing is None:
        return out
    for key, value in existing.answers.items():
        if value == SKIPPED or value is None:
            continue
        out[key] = set(value) if isinstance(value, list) else {str(value)}
    return out


async def _card(request: Request, f: db.JobFilter, seen: list[int], since: datetime, *,
                job: db.Job | None = None, pick_next: bool = True, flash: str = "") -> dict[str, Any]:
    """Everything the card and the counter render: the job (the queue's next
    one unless given), its existing label, the counts."""
    d = _db(request)
    if job is None and pick_next:
        job = await d.next_job_to_label(f, RUBRIC_VERSION, seen)
    existing = await d.get_label(job.id, RUBRIC_VERSION) if job is not None else None
    counts = await d.label_counts(f, RUBRIC_VERSION, seen, since)
    return dict(job=job, existing=existing, existing_summary=summarize(existing.answers) if existing else "",
                checked=checked_values(existing), counts=counts, rubric=RUBRIC, filter=f, flash=flash)


async def _facets(request: Request) -> dict[str, Any]:
    d = _db(request)
    return dict(facets=await d.job_facets(), sites=await d.sites_with_jobs())


@router.get("/label", response_class=HTMLResponse)
async def label_page(request: Request) -> HTMLResponse:
    q = request.query_params
    ctx = await _card(request, label_filter(q), parse_seen(q.get("seen")), parse_since(q.get("since")))
    return _page(request, **ctx, **await _facets(request))


@router.get("/label/next", response_class=HTMLResponse)
async def label_next(request: Request) -> HTMLResponse:
    """The next card for the current filter, as a fragment (plus the counter
    out of band). ``seen`` excludes what this session already showed."""
    q = request.query_params
    ctx = await _card(request, label_filter(q), parse_seen(q.get("seen")), parse_since(q.get("since")))
    return _fragment(request, **ctx)


@router.get("/label/export")
async def label_export(request: Request) -> PlainTextResponse:
    """Every label under the current rubric as JSON Lines, the job's fields
    frozen beside the answers, so the eval runs on a file rather than on
    live rows."""
    rows = await _db(request).export_labels(RUBRIC_VERSION)
    now = datetime.now(timezone.utc)
    stamp = now.isoformat(timespec="seconds")
    lines = [json.dumps({"exported_at": stamp, **row.model_dump(mode="json")}, ensure_ascii=False) for row in rows]
    body = "\n".join(lines) + ("\n" if lines else "")
    filename = f"r3-labels-{RUBRIC_VERSION.replace('/', '-')}-{now:%Y%m%d}.jsonl"
    return PlainTextResponse(body, media_type="application/x-ndjson",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                      "X-R3-Labels": str(len(rows))})


@router.get("/label/{job_id}", response_class=HTMLResponse)
async def label_job(request: Request, job_id: DbId) -> HTMLResponse:
    """One specific posting (re-opening a labeled job, undo): the card shows
    the existing label pre-selected."""
    job = await _db(request).get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    q = request.query_params
    ctx = await _card(request, label_filter(q), parse_seen(q.get("seen")), parse_since(q.get("since")), job=job)
    if _is_htmx(request):
        return _fragment(request, **ctx)
    return _page(request, **ctx, **await _facets(request))


@router.post("/label/{job_id}", response_class=HTMLResponse)
async def label_save(request: Request, job_id: DbId) -> HTMLResponse:
    """Store the answers (overwriting an earlier label, which goes to the
    history table) and answer with the next card for the same filter."""
    d = _db(request)
    job = await d.get_job(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    form = await request.form()
    values = {key: [str(v) for v in form.getlist(key)] for key in form.keys()}
    try:
        answers = parse_answers(values)
        note = clean_note(str(form.get("note") or ""))
    except ValueError as e:
        raise HTTPException(400, str(e))
    label, replaced = await d.save_label(
        job_id, RUBRIC_VERSION, answers, note, labeler=DEFAULT_LABELER,
        session_id=clean_session_id(str(form.get("session_id") or "")),
        seconds_spent=clean_seconds(str(form.get("seconds_spent") or "")))
    log.info("label saved: job %s (%s)%s", job_id, answers.get("relevant"), " replaced" if replaced else "")
    seen = parse_seen(form.get("seen"))
    if job_id not in seen:
        seen.append(job_id)
    flash = f"Saved #{job_id} {job.position} · {summarize(answers)}"
    if replaced:
        flash += " · replaced the earlier label"
    ctx = await _card(request, label_filter(form), seen, parse_since(form.get("since")), flash=flash)
    ctx["saved"] = label
    return _fragment(request, **ctx)
