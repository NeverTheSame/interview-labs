"""The resume library (``/resume``): the job seeker uploads or pastes their
resume, edits it, picks the primary one and downloads it as text or PDF.

Rows live in ``r3_resumes`` (:mod:`r3.db.resumes`); file parsing is
:func:`r3.resume.extract_text`. Every handler takes the account from
:func:`r3.web.auth.current_user` and passes its id to the query, so an id
that belongs to another account answers 404 just like a missing one.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from pathlib import PurePath
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.concurrency import run_in_threadpool

from .limits import UPLOAD_TOO_LARGE_KEY, too_large_message
from r3.resume import ACCEPTED_EXTENSIONS, MAX_UPLOAD_BYTES, ResumeExtractError, extract_text, normalize_text
from r3.events import record as record_event
from r3.resume.parse import looks_like_resume
from r3.resume.pdf import render_pdf

from .auth import current_user
from .inputs import DbId

router = APIRouter(tags=["resume"])

MIN_CHARS = 50
MAX_CHARS = 100_000
MAX_TITLE = 200
# Upload guardrails shared by the Resume page and the Assistant's upload.
EXTRACT_TIMEOUT_S = 20.0
UPLOADS_PER_HOUR = 10

_STATUS = {"unsupported": 415, "too_large": 413}


class _Invalid(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _db(request: Request) -> Any:
    return request.app.state.db


def _render(request: Request, name: str, *, title: str, status_code: int = 200, **ctx: Any) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template(f"{name}.html")
    ctx.setdefault("query", request.query_params)
    return HTMLResponse(tmpl.render(title=title, nav="resume", **ctx), status_code=status_code)


def _redirect(request: Request, path: str) -> RedirectResponse:
    return RedirectResponse(f"{request.scope['root_path']}{path}", status_code=303)


def _check_content(text: str) -> str:
    if len(text) < MIN_CHARS:
        raise _Invalid(f"That is only {len(text)} characters; a resume needs at least {MIN_CHARS}. "
                       "Paste the whole resume.")
    if len(text) > MAX_CHARS:
        raise _Invalid(f"That is {len(text):,} characters; the limit is {MAX_CHARS:,}. "
                       "Trim it to the resume itself.", 413)
    return text


def _clean_title(title: str, fallback: str) -> str:
    title = " ".join(title.split())[:MAX_TITLE]
    return title or fallback


def download_filename(title: str) -> str:
    """An ASCII filename safe for any Content-Disposition parser."""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-._")[:80]
    return f"{stem or 'resume'}.txt"


async def _library(request: Request, *, status_code: int = 200, upload_error: str = "", paste_error: str = "",
                   paste_title: str = "", paste_content: str = "") -> HTMLResponse:
    resumes = await _db(request).list_resumes(current_user(request).id)
    return _render(request, "resume", title="Resume", status_code=status_code, resumes=resumes,
                   upload_error=upload_error, paste_error=paste_error, paste_title=paste_title,
                   paste_content=paste_content, accepted=",".join(ACCEPTED_EXTENSIONS),
                   max_mb=MAX_UPLOAD_BYTES // 1024 // 1024, min_chars=MIN_CHARS, max_chars=MAX_CHARS)


@router.get("/resume", response_class=HTMLResponse)
async def resume_library(request: Request) -> HTMLResponse:
    return await _library(request)


class UploadRefused(Exception):
    """An upload that is not saved; ``message`` says why and what to do."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


async def ingest_upload(request: Request, file: UploadFile | None, title: str = "") -> Any:
    """Every rule an uploaded resume passes before it is saved, for the
    Resume page and the Assistant alike: the hourly limit, the size, type,
    page and time caps, the text bounds and a lenient is-it-a-resume check.
    Returns the saved resume (primary when it is the account's first).
    Both outcomes are audit events: what was saved, or why it was refused."""
    user = current_user(request)
    try:
        resume, size = await _ingest(request, user, file, title)
    except UploadRefused as e:
        record_event("resume.refused", f"Upload refused: {e.message}", actor="user", user_id=user.id,
                     data={"filename": getattr(file, "filename", "") or "", "status": e.status, "reason": e.message})
        raise
    record_event("resume.uploaded", f"You uploaded “{resume.title}”", actor="user", user_id=user.id,
                 subject=f"resume:{resume.id}",
                 data={"title": resume.title, "filename": resume.filename, "bytes": size,
                       "chars": len(resume.content), "primary": resume.is_primary})
    return resume


async def check_hourly_limit(request: Request, user: Any) -> None:
    """Uploads and chat-made resumes share one hourly limit (each is a read)."""
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    if await _db(request).count_recent_uploads(user.id, since) >= UPLOADS_PER_HOUR:
        raise UploadRefused(f"You have uploaded {UPLOADS_PER_HOUR} resumes in the last hour. "
                            "Try again in a while, or edit the one you have.", 429)


def check_text(text: str) -> str:
    """Typed or pasted resume text, normalised and within the bounds."""
    try:
        return _check_content(normalize_text(text))
    except _Invalid as e:
        raise UploadRefused(e.message, e.status) from None


def clean_title(title: str, fallback: str) -> str:
    return _clean_title(title, fallback)


async def _ingest(request: Request, user: Any, file: UploadFile | None, title: str) -> tuple[Any, int]:
    if request.scope.get(UPLOAD_TOO_LARGE_KEY):
        # r3.web.limits refused the body before it was parsed (R3-12).
        raise UploadRefused(too_large_message(), 413)
    if file is None or not file.filename:
        raise UploadRefused("Choose a resume file to upload.", 400)
    await check_hourly_limit(request, user)
    # Read one byte past the limit so an oversized file is rejected without
    # holding more of it than needed.
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    filename = PurePath(file.filename.replace("\\", "/")).name[-255:]
    try:
        # PDF parsing is CPU-bound; keep it off the event loop, and give up
        # on a file that takes too long (the thread finishes on its own).
        raw = await asyncio.wait_for(run_in_threadpool(extract_text, filename, data), EXTRACT_TIMEOUT_S)
        text = _check_content(raw)
    except TimeoutError:
        raise UploadRefused("R3 could not read that file in time. Save it as a smaller PDF, or paste the text "
                            "on the Resume page.", 422) from None
    except ResumeExtractError as e:
        raise UploadRefused(e.message, _STATUS.get(e.kind, 400)) from None
    except _Invalid as e:
        raise UploadRefused(e.message, e.status) from None
    if not looks_like_resume(text):
        raise UploadRefused("This doesn't look like a resume: R3 found no contact details, dates, job titles or "
                            "resume sections in it. Upload your resume file.", 422)
    resume = await _db(request).create_resume(
        user.id, title=_clean_title(title, PurePath(filename).stem or "Uploaded resume"),
        source="upload", filename=filename, content=text)
    return resume, len(data)


@router.post("/resume/upload")
async def resume_upload(request: Request, file: UploadFile | None = File(None), title: str = Form("")) -> Response:
    try:
        resume = await ingest_upload(request, file, title)
    except UploadRefused as e:
        return await _library(request, status_code=e.status, upload_error=e.message)
    return _redirect(request, f"/resume/{resume.id}")


@router.post("/resume/paste")
async def resume_paste(request: Request, title: str = Form(""), content: str = Form("")) -> Response:
    try:
        text = _check_content(normalize_text(content))
    except _Invalid as e:
        return await _library(request, status_code=e.status, paste_error=e.message,
                              paste_title=title, paste_content=content)
    user = current_user(request)
    resume = await _db(request).create_resume(user.id, title=_clean_title(title, "Pasted resume"), source="paste",
                                              content=text)
    record_event("resume.created", f"You pasted “{resume.title}”", actor="user", user_id=user.id,
                 subject=f"resume:{resume.id}", data={"source": "paste", "chars": len(text)})
    return _redirect(request, f"/resume/{resume.id}")


async def _owned(request: Request, resume_id: int) -> Any:
    resume = await _db(request).get_resume(current_user(request).id, resume_id)
    if resume is None:
        raise HTTPException(404, "resume not found")
    return resume


def missing_items(resume: Any) -> list[str]:
    """What a read resume still lacks of the builder's hard requirements
    (dates on every role, a profile link), in words; [] when complete or
    never read (r3.resume.parse via the Assistant's upload)."""
    from r3.resume.builder import ResumeDraft, open_requirements

    if not resume.structure:
        return []
    draft = ResumeDraft(**resume.structure.get("draft", {}))
    out = []
    for need in open_requirements(draft):
        if need == "dates":
            names = ", ".join(" ".join((j.title or j.company or "a role").split())
                              for j in draft.experience if not j.start.strip() or not j.end.strip())
            out.append(f"dates for {names}")
        else:
            out.append("a profile link")
    return out


def _detail(request: Request, resume: Any, *, status_code: int = 200, error: str = "",
            form_title: str | None = None, form_content: str | None = None) -> HTMLResponse:
    return _render(request, "resume_detail", title=resume.title, status_code=status_code, resume=resume,
                   missing=missing_items(resume),
                   error=error, form_title=resume.title if form_title is None else form_title,
                   form_content=resume.content if form_content is None else form_content)


@router.get("/resume/{resume_id:int}", response_class=HTMLResponse)
async def resume_detail(request: Request, resume_id: DbId) -> HTMLResponse:
    return _detail(request, await _owned(request, resume_id))


@router.post("/resume/{resume_id:int}")
async def resume_save(request: Request, resume_id: DbId, title: str = Form(""), content: str = Form("")) -> Response:
    resume = await _owned(request, resume_id)
    try:
        clean_title = _clean_title(title, "")
        if not clean_title:
            raise _Invalid("Give the resume a title.")
        text = _check_content(normalize_text(content))
    except _Invalid as e:
        return _detail(request, resume, status_code=e.status, error=e.message, form_title=title, form_content=content)
    updated = await _db(request).update_resume(current_user(request).id, resume_id, title=clean_title, content=text)
    if updated is None:  # deleted in another tab between the read and the write
        raise HTTPException(404, "resume not found")
    record_event("resume.edited", f"You edited “{clean_title}”", actor="user", subject=f"resume:{resume_id}",
                 data={"title": clean_title, "chars_before": len(resume.content), "chars_after": len(text)})
    return _redirect(request, f"/resume/{resume_id}")


@router.post("/resume/{resume_id:int}/primary")
async def resume_primary(request: Request, resume_id: DbId) -> Response:
    if not await _db(request).set_primary_resume(current_user(request).id, resume_id):
        raise HTTPException(404, "resume not found")
    record_event("resume.primary", f"You made resume {resume_id} your primary resume", actor="user",
                 subject=f"resume:{resume_id}")
    return _redirect(request, "/resume")


@router.post("/resume/{resume_id:int}/delete")
async def resume_delete(request: Request, resume_id: DbId) -> Response:
    if not await _db(request).delete_resume(current_user(request).id, resume_id):
        raise HTTPException(404, "resume not found")
    record_event("resume.deleted", f"You deleted resume {resume_id}", actor="user", subject=f"resume:{resume_id}")
    return _redirect(request, "/resume")


@router.get("/resume/{resume_id:int}/download")
async def resume_download(request: Request, resume_id: DbId) -> Response:
    resume = await _owned(request, resume_id)
    name = download_filename(resume.title)
    return Response(resume.content + "\n", media_type="text/plain; charset=utf-8", headers={
        "Content-Disposition": f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}",
        "X-Content-Type-Options": "nosniff",
    })


@router.get("/resume/{resume_id:int}/pdf")
async def resume_pdf(request: Request, resume_id: DbId) -> Response:
    resume = await _owned(request, resume_id)
    # Layout and font subsetting are CPU-bound; keep them off the event loop.
    data = await run_in_threadpool(render_pdf, resume.title, resume.content)
    name = download_filename(resume.title).removesuffix(".txt") + ".pdf"
    return Response(data, media_type="application/pdf", headers={
        "Content-Disposition": f"attachment; filename=\"{name}\"; filename*=UTF-8''{quote(name)}",
        "X-Content-Type-Options": "nosniff",
    })
