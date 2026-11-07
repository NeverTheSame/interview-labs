"""The resume builder (``/resume/build``): a job seeker without a resume
answers the assistant's questions in a chat and watches the resume take
shape in the panel beside it, then saves it to the resume library.

The interview itself is :mod:`r3.resume.builder`; the interviewer (LLM or
scripted) is chosen at startup and lives on ``app.state.resume_interviewer``.
The in-progress draft is one row per account in ``r3_resume_drafts``
(:mod:`r3.db.resume_drafts`), so a reload or a new sign-in continues where
the person stopped. Every handler scopes by :func:`r3.web.auth.current_user`.

This router is included before the resume library's, so ``/resume/build``
is never read as ``/resume/{id}``.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from pydantic import BaseModel, ValidationError

from r3.events import record as record_event
from r3.resume.builder import ChatEntry, Interviewer, ResumeDraft, ScriptedInterviewer, completeness, render_resume_text

from .auth import current_user
from .routes_resume import MAX_TITLE, MIN_CHARS

log = logging.getLogger(__name__)

router = APIRouter(tags=["resume"])

MAX_ANSWER = 8000


class AnswerIn(BaseModel):
    answer: str = ""


def _interviewer(request: Request) -> Interviewer:
    return request.app.state.resume_interviewer


async def _load(request: Request) -> tuple[ResumeDraft, list[ChatEntry]]:
    """The account's saved draft and conversation, or a fresh one opened
    with the interviewer's first question."""
    stored = await request.app.state.db.get_resume_draft(current_user(request).id)
    if stored is not None:
        try:
            draft = ResumeDraft.model_validate(stored.draft)
            history = [ChatEntry.model_validate(e) for e in stored.history]
            if history:
                return draft, history
        except ValidationError as e:  # an older shape: start over rather than 500
            log.warning("discarding unreadable resume draft: %s", e)
    opening = _interviewer(request).opening(ResumeDraft())
    return opening.draft, [ChatEntry(role="assistant", text=opening.question, topic=opening.topic)]


def _progress(draft: ResumeDraft) -> dict[str, Any]:
    items = completeness(draft)
    return {"done": sum(ok for _, ok in items), "total": len(items),
            "items": [{"section": name, "done": ok} for name, ok in items]}


def _fragment(request: Request, name: str, **ctx: Any) -> str:
    return request.app.state.templates.get_template(f"{name}.html").render(**ctx)


async def _page(request: Request, *, status_code: int = 200, error: str = "", title: str = "") -> HTMLResponse:
    draft, history = await _load(request)
    tmpl = request.app.state.templates.get_template("resume_build.html")
    last = history[-1] if history else None
    return HTMLResponse(tmpl.render(
        title="Build your resume", nav="resume", query={}, draft=draft, history=history,
        progress=_progress(draft), scripted=isinstance(_interviewer(request), ScriptedInterviewer),
        done=last is not None and last.done,
        error=error, save_title=title, max_answer=MAX_ANSWER, max_title=MAX_TITLE), status_code=status_code)


@router.get("/resume/build", response_class=HTMLResponse)
async def build_page(request: Request) -> HTMLResponse:
    return await _page(request)


def _error(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@router.post("/resume/build/answer")
async def build_answer(request: Request, body: AnswerIn) -> JSONResponse:
    answer = body.answer.strip()
    if not answer:
        return _error("Type an answer first (or \"skip\" to move on).", 400)
    if len(answer) > MAX_ANSWER:
        return _error(f"That is {len(answer):,} characters; one answer can be at most {MAX_ANSWER:,}.", 413)
    user = current_user(request)
    draft, history = await _load(request)
    try:
        turn = await _interviewer(request).next_turn(draft, history, answer)
    except Exception as e:  # LLMInterviewer never raises; an injected one might
        log.warning("resume builder turn failed: %s: %s", type(e).__name__, e)
        return _error("The interviewer failed on that answer. Your answer is back in the box; try again.", 502)
    history = [*history, ChatEntry(role="user", text=answer),
               ChatEntry(role="assistant", text=turn.question, topic=turn.topic, done=turn.done)]
    await request.app.state.db.save_resume_draft(
        user.id, draft=turn.draft.model_dump(), history=[e.model_dump() for e in history])
    progress = _progress(turn.draft)
    return JSONResponse({
        "question": turn.question, "done": turn.done, "note": turn.note, "progress": progress,
        "draft_html": _fragment(request, "_resume_preview", draft=turn.draft),
        "progress_html": _fragment(request, "_resume_progress", progress=progress),
    })


@router.post("/resume/build/save")
async def build_save(request: Request, title: str = Form("")) -> Response:
    user = current_user(request)
    draft, _ = await _load(request)
    text = render_resume_text(draft)
    if len(text) < MIN_CHARS:
        return await _page(request, status_code=400, title=title, error=(
            f"The resume is only {len(text)} characters so far; a resume needs at least {MIN_CHARS}. "
            "Answer a few more questions first."))
    # Hard requirements: a verifiable profile link, and dates on every role.
    needs = []
    if not draft.contact.links:
        needs.append("at least one profile link — LinkedIn, GitHub or your portfolio. Employers check you up "
                     "before they reply, and a real link is what makes the resume verifiable")
    dateless = [j for j in draft.experience if not j.start.strip() or not j.end.strip()]
    if dateless:
        names = ", ".join(" ".join((j.title or j.company or "a role").split()) for j in dateless)
        needs.append(f"start and end dates for: {names}")
    if needs:
        return await _page(request, status_code=400, title=title, error=(
            "Before saving, this resume needs " + "; and ".join(needs)
            + ". Type it in the interview — I'll fold it in — and save again."))
    clean = " ".join(title.split())[:MAX_TITLE]
    if not clean:
        clean = " — ".join(p for p in (draft.full_name.strip(), draft.headline.strip()) if p)[:MAX_TITLE] \
            or "Built resume"
    resume = await request.app.state.db.create_resume(user.id, title=clean, source="builder", content=text)
    await request.app.state.db.delete_resume_draft(user.id)
    record_event("resume.created", f"You saved “{clean}” from the builder", actor="user", user_id=user.id,
                 subject=f"resume:{resume.id}",
                 data={"source": "builder", "roles": len(draft.experience), "links": len(draft.contact.links),
                       "chars": len(text)})
    return RedirectResponse(f"{request.scope['root_path']}/resume/{resume.id}", status_code=303)


@router.post("/resume/build/reset")
async def build_reset(request: Request) -> Response:
    await request.app.state.db.delete_resume_draft(current_user(request).id)
    return RedirectResponse(f"{request.scope['root_path']}/resume/build", status_code=303)
