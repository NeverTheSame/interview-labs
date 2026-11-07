"""Resume upload from the Assistant chat (JSON under ``/api/resume``).

``POST /upload`` saves a file with the Resume page's rules
(:func:`r3.web.routes_resume.ingest_upload`). ``POST /{id}/read`` reads it
into the builder's structure (:mod:`r3.resume.parse`, ``app.state.resume_parser``)
and keeps R3's clean layout when it is proved faithful. ``/{id}/followup``
and ``/{id}/answer`` ask for the builder's hard requirements, dates on every
role and a profile link (:mod:`r3.resume.followup`), and fold the answers in.
``POST /from-text`` makes a resume from what the person wrote in the chat
(:mod:`r3.resume.background`) and answers like a read.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from r3.events import record as record_event
from r3.resume.background import BackgroundRefused, read_background
from r3.resume.builder import ResumeDraft, completeness, missing_requirements, render_resume_text
from r3.resume.followup import FollowupTurn, answer_followup, next_followup
from r3.resume.parse import ParsedResume

from .auth import User, current_user
from .inputs import DbId
from .routes_resume import UploadRefused, check_hourly_limit, check_text, clean_title, ingest_upload
from .routes_resume_build import MAX_ANSWER

router = APIRouter(prefix="/api/resume", tags=["resume-chat"])
log = logging.getLogger(__name__)

READ_FAILED = "Saved as uploaded; I couldn't read its structure. It still works for job matching."


def init_state(app: FastAPI) -> None:
    app.state.resume_reads = {}


async def cancel_reads(app: FastAPI) -> None:
    tasks = list(getattr(app.state, "resume_reads", {}).values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


class AnswerIn(BaseModel):
    answer: str = ""


class FromTextIn(BaseModel):
    text: str = ""
    title: str = ""


def _followup(draft: ResumeDraft, turn: FollowupTurn | None = None) -> dict[str, Any]:
    turn = turn or next_followup(draft)
    return {"topic": turn.topic, "question": turn.question, "done": turn.done, "note": turn.note,
            "changed": turn.changed, "off_topic": turn.off_topic, "open": missing_requirements(turn.draft),
            "checklist": [{"section": name, "ok": ok} for name, ok in completeness(turn.draft)]}


def _card(request: Request, resume: Any) -> dict[str, Any]:
    s = resume.structure or {}
    draft = ResumeDraft(**s.get("draft", {}))
    return {
        "resume_id": resume.id, "title": resume.title, "primary": resume.is_primary,
        "url": f"/resume/{resume.id}",  # the page adds its base path
        "read": bool(s), "faithful": bool(s.get("faithful")), "engine": s.get("engine", ""),
        "is_resume": s.get("is_resume", True),
        "name": draft.full_name, "headline": draft.headline, "email": draft.contact.email,
        "phone": draft.contact.phone, "location": draft.contact.location, "links": draft.contact.links,
        "roles": [{"title": j.title, "company": j.company, "start": j.start, "end": j.end, "bullets": len(j.bullets)}
                  for j in draft.experience],
        "skills": draft.skills[:15], "skills_count": len(draft.skills), "education": draft.education[:3],
        **_followup(draft),
    }


def _content(structure: dict[str, Any], draft: ResumeDraft) -> str:
    """The text to keep: the clean layout when the read was faithful, else
    the original with the seeker's answers listed under it."""
    if structure.get("faithful"):
        return render_resume_text(draft)
    added = structure.get("added") or []
    source = structure.get("source", "")
    return source + ("\n\nAdded in R3\n" + "\n".join(f"- {a}" for a in added) if added else "")


async def _mine(request: Request, user: User, resume_id: int) -> Any:
    resume = await request.app.state.db.get_resume(user.id, resume_id)
    if resume is None:
        raise HTTPException(404, "resume not found")
    return resume


@router.post("/upload")
async def upload(request: Request, file: UploadFile | None = File(None), title: str = Form(""),
                 user: User = Depends(current_user)) -> dict[str, Any]:
    try:
        resume = await ingest_upload(request, file, title)
    except UploadRefused as e:
        raise HTTPException(e.status, e.message) from None
    return {"resume_id": resume.id, "title": resume.title, "primary": resume.is_primary, "chars": len(resume.content)}


def _structure(parsed: ParsedResume, source: str, *, keep_source: bool = False) -> dict[str, Any]:
    structure: dict[str, Any] = {
        "draft": parsed.draft.model_dump(), "engine": parsed.engine, "model": parsed.model,
        "faithful": parsed.faithful, "is_resume": parsed.is_resume, "added": [],
        "read_at": datetime.now(timezone.utc).isoformat(),
    }
    if keep_source or not parsed.faithful:
        structure["source"] = source
    return structure


def _record_read(user_id: int, resume: Any, parsed: ParsedResume) -> None:
    d = parsed.draft
    record_event("resume.read", f"R3 read “{resume.title}”: {len(d.experience)} roles, {len(d.contact.links)} links, "
                 f"{len(d.skills)} skills" + ("" if parsed.faithful else "; kept your original text"),
                 actor="ai", user_id=user_id, subject=f"resume:{resume.id}",
                 data={"engine": parsed.engine, "model": parsed.model, "faithful": parsed.faithful,
                       "is_resume": parsed.is_resume, "roles": len(d.experience), "links": len(d.contact.links),
                       "skills": len(d.skills),
                       "roles_read": [{"title": j.title, "company": j.company, "start": j.start, "end": j.end}
                                      for j in d.experience]})


async def _read(app: FastAPI, user_id: int, resume: Any) -> Any:
    parsed = await app.state.resume_parser.parse(resume.content)
    _record_read(user_id, resume, parsed)
    return await app.state.db.set_resume_structure(user_id, resume.id, structure=_structure(parsed, resume.content),
                                                   content=parsed.text)


@router.post("/from-text")
async def from_text(request: Request, body: FromTextIn, user: User = Depends(current_user)) -> dict[str, Any]:
    """A resume from what the person wrote in the chat: saved only when R3
    could read it, with the shared hourly limit and text bounds."""
    try:
        text = check_text(body.text)
        await check_hourly_limit(request, user)
        parsed = await read_background(text, parser=request.app.state.resume_parser,
                                       interviewer=request.app.state.resume_interviewer)
    except (UploadRefused, BackgroundRefused) as e:
        status = getattr(e, "status", 422)
        record_event("resume.refused", f"Resume from the chat refused: {e.message}", actor="user", user_id=user.id,
                     data={"source": "chat", "status": status, "reason": e.message, "chars": len(body.text)})
        raise HTTPException(status, e.message) from None
    name = parsed.draft.full_name
    db = request.app.state.db
    resume = await db.create_resume(user.id, title=clean_title(body.title, f"{name} resume" if name else
                                                               "Resume from the chat"),
                                    source="chat", content=parsed.text)
    record_event("resume.created", f"You described your background; R3 made “{resume.title}”", actor="user",
                 user_id=user.id, subject=f"resume:{resume.id}",
                 data={"source": "chat", "chars": len(text), "primary": resume.is_primary})
    _record_read(user.id, resume, parsed)
    saved = await db.set_resume_structure(user.id, resume.id, structure=_structure(parsed, text, keep_source=True),
                                          content=parsed.text)
    return _card(request, saved or resume)


@router.post("/{resume_id}/read")
async def read(request: Request, resume_id: DbId, user: User = Depends(current_user)) -> dict[str, Any]:
    resume = await _mine(request, user, resume_id)
    if resume.structure:
        return _card(request, resume)
    reads: dict[int, asyncio.Task[Any]] = request.app.state.resume_reads
    task = reads.get(resume.id)
    if task is None:
        task = asyncio.ensure_future(_read(request.app, user.id, resume))
        reads[resume.id] = task
        task.add_done_callback(lambda _: reads.pop(resume.id, None))
    try:
        # Shielded: a seeker who leaves does not cancel a read another tab waits on.
        saved = await asyncio.shield(task)
    except Exception as e:
        log.warning("resume read failed: %s: %s", type(e).__name__, e)
        return {**_card(request, resume), "message": READ_FAILED}
    return _card(request, saved or resume)


@router.get("/{resume_id}/followup")
async def followup(request: Request, resume_id: DbId, user: User = Depends(current_user)) -> dict[str, Any]:
    return _card(request, await _mine(request, user, resume_id))


@router.post("/{resume_id}/answer")
async def answer(request: Request, resume_id: DbId, body: AnswerIn,
                 user: User = Depends(current_user)) -> dict[str, Any]:
    text = body.answer.strip()
    if not text:
        raise HTTPException(400, "Type an answer first, or \"skip\".")
    if len(text) > MAX_ANSWER:
        raise HTTPException(413, f"That is {len(text):,} characters; one answer can be at most {MAX_ANSWER:,}.")
    resume = await _mine(request, user, resume_id)
    if not resume.structure:
        raise HTTPException(409, "R3 has not read this resume yet.")
    structure = dict(resume.structure)
    turn = answer_followup(ResumeDraft(**structure.get("draft", {})), text)
    structure["draft"] = turn.draft.model_dump()
    if turn.changed and not turn.changed.startswith("skipped"):
        structure["added"] = [*(structure.get("added") or []), turn.changed]
    await request.app.state.db.set_resume_structure(user.id, resume.id, structure=structure,
                                                    content=_content(structure, turn.draft))
    record_event("resume.answer", f"You answered a follow-up: {turn.changed or text}", actor="user",
                 user_id=user.id, subject=f"resume:{resume.id}", data={"answer": text, "changed": turn.changed})
    return _followup(turn.draft, turn)
