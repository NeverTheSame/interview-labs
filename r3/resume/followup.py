"""The follow-up questions the Assistant asks after reading an uploaded
resume: only the builder's hard requirements (OQ-2/3), dates on every role
and at least one profile link, asked one at a time. Deterministic, no model
call, so every answer is instant.

Answers are read the way people write them ("2021 - present", "Jul 2021 –
now", "since 2020", "2019, 2022"). A message that is not an answer but a
question back ("what do you need?") is marked ``off_topic`` so the chat
hands it to the assistant instead of repeating the question.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .builder import LINKS_QUESTION, Job, ResumeDraft, _dates_from, is_skip, missing_requirements
from .parse import _DATE, _RANGE, _links

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PRESENT = re.compile(r"(?i)^(present|current|now|today)$")
_SINCE = re.compile(rf"(?i)\bsince\s+(?P<start>{_DATE})\b")
_QUESTION = re.compile(r"(?i)^(what|why|how|who|which|where|when|can|could|do|does|is|are|should|would|will|"
                       r"explain|help|huh)\b")
DATES_HELP = "I couldn't find a start and end date in that."
LINK_HELP = "That doesn't look like a link. Paste the full address, e.g. linkedin.com/in/yourname."


class FollowupTurn(BaseModel):
    """The draft after an answer, the next question ("" when done), a note
    when the answer could not be used, what changed, and ``off_topic`` when
    the message was a question back rather than an answer."""

    draft: ResumeDraft
    topic: str = ""
    question: str = ""
    done: bool = False
    note: str = ""
    changed: str = ""
    off_topic: bool = False


def _role(job: Job) -> str:
    title = " ".join(job.title.split())
    company = " ".join(job.company.split())
    if title and company:
        return f"{title} at {company}"
    return title or company or "that role"


def dates_question(draft: ResumeDraft) -> str:
    job = next(j for j in draft.experience if not j.start.strip() or not j.end.strip())
    if job.start.strip():
        return f"You started as {_role(job)} in {job.start}. When did it end? “Present” is fine if you're still there."
    return f"When did you start and finish as {_role(job)}? Something like “Jul 2021 – present” works."


def _question(topic: str, draft: ResumeDraft) -> str:
    return dates_question(draft) if topic == "dates" else LINKS_QUESTION


def next_followup(draft: ResumeDraft, *, note: str = "", changed: str = "") -> FollowupTurn:
    topics = missing_requirements(draft)
    if not topics:
        return FollowupTurn(draft=draft, done=True, note=note, changed=changed)
    return FollowupTurn(draft=draft, topic=topics[0], question=_question(topics[0], draft), note=note,
                        changed=changed)


def _readable(start: str, end: str) -> bool:
    return bool(_YEAR.search(start)) and bool(_YEAR.search(end) or _PRESENT.match(end.strip()))


def _read_dates(text: str, job: Job) -> tuple[str, str] | None:
    """(start, end) from an answer, or None when it holds no usable dates."""
    if m := _RANGE.search(text):
        return m.group("start"), m.group("end")
    if m := _SINCE.search(text):
        return m.group("start"), "present"
    trial = job.model_copy()
    _dates_from(text, trial)  # the builder's "start, end" form, and a lone end once the start is known
    return (trial.start, trial.end) if _readable(trial.start, trial.end) else None


def _looks_like_a_question(text: str) -> bool:
    return "?" in text or bool(_QUESTION.match(text.strip()))


def answer_followup(draft: ResumeDraft, answer: str) -> FollowupTurn:
    """File ``answer`` against the first open requirement and return the
    next question. An answer that cannot be read leaves the draft as it was;
    a question back is marked off-topic; "skip" records the requirement as
    skipped, as the builder does."""
    draft = draft.model_copy(deep=True)
    topics = missing_requirements(draft)
    if not topics:
        return next_followup(draft)
    topic, text = topics[0], answer.strip()
    if is_skip(text):
        draft.skipped.append(topic)
        return next_followup(draft, changed=f"skipped the {'dates' if topic == 'dates' else 'link'}")
    if topic == "links":
        found = [link for link in _links(text) if link not in draft.contact.links]
        if not found:
            if _looks_like_a_question(text):
                return FollowupTurn(draft=draft, topic=topic, question=LINKS_QUESTION, off_topic=True)
            return FollowupTurn(draft=draft, topic=topic, question=LINKS_QUESTION, note=LINK_HELP)
        draft.contact.links.extend(found)
        return next_followup(draft, changed="Profile link: " + ", ".join(found))
    job = next(j for j in draft.experience if not j.start.strip() or not j.end.strip())
    dates = _read_dates(text, job)
    if dates is None:
        if _looks_like_a_question(text):
            return FollowupTurn(draft=draft, topic=topic, question=dates_question(draft), off_topic=True)
        return FollowupTurn(draft=draft, topic=topic, question=dates_question(draft), note=DATES_HELP)
    job.start, job.end = dates
    return next_followup(draft, changed=f"{_role(job)}: {job.start} – {job.end}")
