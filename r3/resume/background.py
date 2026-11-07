"""A resume from what the person wrote in the Assistant chat.

Resume-shaped text (a pasted resume or LinkedIn profile, one item per line)
goes through the reader like an upload (:mod:`r3.resume.parse`) and keeps
the person's own words. A description in prose ("I've been a backend
developer at Stripe since 2021...") has no lines to label, so the builder's
model extracts the draft from it (:meth:`r3.resume.builder.LLMInterviewer.extract`),
held to what was written. The saved text is then R3's layout of that draft.
"""

from __future__ import annotations

from typing import Any

from .builder import render_resume_text
from .parse import ParsedResume

# A pasted resume has an item per line; a description in prose a few long lines.
RESUME_SHAPED_LINES = 6

NO_MODEL = ("Turning a description into a resume needs the AI model, and none is set up. Use "
            "[Build your resume](/resume/build), where R3 asks a few questions, or the \"Upload resume\" button.")
FAILED = ("I couldn't turn that into a resume this time. Try again, or use [Build your resume](/resume/build), "
          "where R3 asks a few questions, or the \"Upload resume\" button.")
NOT_BACKGROUND = ("That doesn't read like your background: I found no name, job or skills in it. Tell me your "
                  "jobs (title, company, years), your skills and your education, and ask again.")


class BackgroundRefused(Exception):
    """Nothing is saved; ``message`` says why and what to do."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def resume_shaped(text: str) -> bool:
    return sum(1 for line in text.splitlines() if line.strip()) >= RESUME_SHAPED_LINES


async def read_background(text: str, *, parser: Any, interviewer: Any) -> ParsedResume:
    """The draft and the text to save for ``text``, or :class:`BackgroundRefused`."""
    if resume_shaped(text):
        parsed: ParsedResume = await parser.parse(text)
        if parsed.draft.experience:
            return parsed
    extract = getattr(interviewer, "extract", None)
    if extract is None:
        raise BackgroundRefused(NO_MODEL)
    try:
        draft = await extract(text)
    except Exception as e:  # a failed or rejected extraction saves nothing
        raise BackgroundRefused(FAILED) from e
    if not (draft.full_name or draft.experience or draft.skills):
        raise BackgroundRefused(NOT_BACKGROUND)
    return ParsedResume(draft=draft, engine="background", model=getattr(interviewer, "model_id", ""),
                        faithful=True, text=render_resume_text(draft))
