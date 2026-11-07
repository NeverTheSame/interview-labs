"""The "About this role" overview a regular account sees instead of the
posting (OQ-6). R3 is the single place a job seeker works the owner's job
bank, so the raw posting and anything that leads to it (its URL, the board
it came from) stay hidden; two or three sentences, the stack, the pay, the
location and whether the job is live are the free teaser.

:class:`LLMSummarizer` asks a model for the overview with a structured-output
call. :func:`extract_overview` needs no model: the posting's first sentences.
It answers when no model key is configured, when the model call fails or
times out, and in tests. The extract never shows more than half of a
posting's sentences (at most two), so a short posting is not handed over
whole. Either way every link is taken out in code (:mod:`r3.redact`),
because a model told not to write URLs still sometimes does: a sentence
carrying one is dropped. HTML postings are read as text first.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from r3.db.jobs import Job
from r3.redact import has_link as _has_link
from r3.redact import strip_links
from r3.text import plain_description, truncate_description

__all__ = ["ExtractiveSummarizer", "LLMSummarizer", "Overview", "Summarizer", "build_summarizer",
           "extract_overview", "strip_links"]

log = logging.getLogger(__name__)

MAX_OVERVIEW_CHARS = 600
MAX_SENTENCES = 3
# The extract is the posting's own words: at most this many sentences, and
# never more than half of them.
MAX_EXTRACT_SENTENCES = 2
# The head of a posting is what an overview needs; the cap keeps free-tier
# prompts cheap.
PROMPT_DESCRIPTION_CHARS = 6_000
# The seeker is waiting on one paragraph, so the extract answers when the
# model is slower than this.
LLM_TIMEOUT_S = 60.0
# A model answer shorter than this (after links are dropped) is not an overview.
MIN_LLM_CHARS = 40

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")


def _sentences(text: str) -> list[str]:
    """The prose sentences of a posting. A short line without closing
    punctuation is a heading or a bullet fragment, not a sentence."""
    out: list[str] = []
    for line in text.splitlines():
        line = " ".join(line.split()).lstrip("-•*·–— ")
        if not line:
            continue
        if not line.endswith((".", "!", "?", ":")) and len(line.split()) < 6:
            continue
        out.extend(s.strip() for s in _SENTENCE_END.split(line) if s.strip())
    return out


def _clip(text: str, limit: int = MAX_OVERVIEW_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[:limit - 1]
    cut = head.rfind(" ")
    return (head[:cut] if cut > limit // 2 else head).rstrip(" ,;:") + "…"


def _link_free(sentences: list[str], limit: int) -> str:
    """Up to ``limit`` of ``sentences``, skipping any that carries a link."""
    kept = [s for s in sentences if not _has_link(s)][:limit]
    return _clip(strip_links(" ".join(kept)))


class Overview(BaseModel):
    """The overview text and how it was made: ``engine`` is ``"llm"`` (worth
    caching) or ``"extract"`` (the fallback, recomputed every time)."""

    text: str
    engine: Literal["llm", "extract"]
    model: str = ""


def extract_overview(job: Job) -> Overview:
    """The posting's first sentences without links (at most two, never more
    than half of them), or, for a posting too short to excerpt, one sentence
    built from the job's own fields."""
    sentences = _sentences(plain_description(job.description))
    text = _link_free(sentences, min(MAX_EXTRACT_SENTENCES, len(sentences) // 2))
    if not text:
        where = f", {job.location}" if job.location else ""
        text = f"{job.position}" + (f" at {job.company}" if job.company else "") + f"{where}."
        if job.tech_tags:
            text += f" The stack includes {', '.join(dict.fromkeys(job.tech_tags))}."
        text = _clip(strip_links(text))
    return Overview(text=text, engine="extract")


class Summarizer(Protocol):
    def uses_model(self) -> bool: ...

    async def overview(self, job: Job) -> Overview: ...


class ExtractiveSummarizer:
    """The overview without a model (tests, demos, no key)."""

    def uses_model(self) -> bool:
        return False

    async def overview(self, job: Job) -> Overview:
        return extract_overview(job)


INSTRUCTIONS = """\
You write the short overview a job seeker reads before deciding whether a job deserves a closer look.

Write two or three plain sentences: what the company does, what the person in this role will do, and the core stack
or requirements. Use only facts stated in the posting; do not guess.

NEVER include a URL, a web address, an email address, a phone number, how or where to apply, or the name of a job
board, recruiter or agency. No Markdown, no lists, no headings."""


class _ModelOutput(BaseModel):
    summary: str = Field(description="Two or three plain sentences about the role")


def _prompt(job: Job) -> str:
    tags = ", ".join(job.tech_tags) or "-"
    description = strip_links(truncate_description(plain_description(job.description), PROMPT_DESCRIPTION_CHARS))
    return (f"JOB POSTING\nPosition: {job.position}\nCompany: {job.company or '-'}\nLocation: {job.location or '-'}\n"
            f"Stack: {tags}\n\n{description or '(no description)'}")


class LLMSummarizer:
    """The overview from a Pydantic AI structured-output call. ``model`` is a
    ready model (tests pass ``TestModel``/``FunctionModel``) or a zero-arg
    factory resolved per call, so a key saved in Settings is used without a
    restart; a factory raising ``ValueError`` means no key."""

    def __init__(self, model: Model | Callable[[], Model], *, timeout: float = LLM_TIMEOUT_S) -> None:
        self._model = model
        self.timeout = timeout
        self._agent: Agent[None, _ModelOutput] = Agent(output_type=_ModelOutput, instructions=INSTRUCTIONS,
                                                      defer_model_check=True, name="r3-job-overview", retries=1)

    def _resolve(self) -> Model:
        return self._model() if callable(self._model) and not isinstance(self._model, Model) else self._model

    def uses_model(self) -> bool:
        """False when no model key is configured (the extract answers)."""
        try:
            self._resolve()
        except ValueError:
            return False
        return True

    async def overview(self, job: Job) -> Overview:
        try:
            model = self._resolve()
        except ValueError as e:
            log.info("Job overview without a model: %s", e)
            return extract_overview(job)
        try:
            result = await asyncio.wait_for(self._agent.run(_prompt(job), model=model), self.timeout)
            out: _ModelOutput = result.output
            text = _link_free(_sentences(out.summary), MAX_SENTENCES)
            if len(text) < MIN_LLM_CHARS:
                raise ValueError("model returned no usable overview")
        except Exception as e:  # any model failure falls back; the click must not be lost
            log.warning("Job overview model call failed (%s); using the extract", type(e).__name__)
            return extract_overview(job)
        return Overview(text=text, engine="llm", model=str(getattr(model, "model_name", "") or ""))


def build_summarizer(settings: Callable[[], Any]) -> Summarizer:
    """The app's summarizer: the model engine over the live settings (it
    falls back to the extract per call when no provider has a key)."""
    from r3.agent.models import build_model
    from r3.agent.service import task_model_id

    def model() -> Model:
        s = settings()
        if not s.available_models():
            raise ValueError("no chat provider has a key")
        return build_model(task_model_id(s), s)

    return LLMSummarizer(model)
