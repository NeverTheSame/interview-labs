"""The resume builder: an interview that turns a job seeker's answers into a
resume, one question at a time (``/resume/build``).

:class:`ResumeDraft` is the structured resume the interview fills in;
:func:`render_resume_text` turns it into the plain text the resume library
stores, and :func:`completeness` drives the progress checklist.

Two interviewers implement :class:`Interviewer`:

* :class:`LLMInterviewer` asks a model for the next turn (structured output),
  so it can follow up, press for numbers and sort a pasted old resume.
* :class:`ScriptedInterviewer` is deterministic and needs no model: a fixed
  question list whose answers are filed verbatim. It runs when no model key
  is configured, in tests, and for any LLM turn that fails, so an answer is
  never lost to a timeout.

Neither invents facts: the scripted one only copies what was typed, and
the LLM turn is rejected (and replayed through the script) when it adds a
number the person never wrote.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class Contact(BaseModel):
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)


class Job(BaseModel):
    title: str = ""
    company: str = ""
    location: str = ""
    start: str = ""
    end: str = ""
    bullets: list[str] = Field(default_factory=list)


class ResumeDraft(BaseModel):
    """A resume in progress. Every field is optional so a half-finished
    interview is still a valid draft."""

    full_name: str = ""
    contact: Contact = Field(default_factory=Contact)
    headline: str = Field("", description="The target role, e.g. 'Senior Data Analyst'")
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[Job] = Field(default_factory=list, description="Most recent job first")
    education: list[str] = Field(default_factory=list, description="One line per degree or course")
    certifications: list[str] = Field(default_factory=list)
    extras: list[str] = Field(default_factory=list, description="Languages, volunteering, awards, ...")
    skipped: list[str] = Field(default_factory=list,
                               description="Required follow-ups (dates, links) the person said they cannot give")


class ChatEntry(BaseModel):
    """One line of the interview. ``topic`` is the scripted step an
    assistant question belongs to; LLM questions leave it empty."""

    role: Literal["assistant", "user"]
    text: str
    topic: str = ""
    done: bool = False


class BuilderTurn(BaseModel):
    draft: ResumeDraft
    question: str
    done: bool = False
    note: str = ""
    topic: str = ""


class Interviewer(Protocol):
    async def next_turn(self, draft: ResumeDraft, history: Sequence[ChatEntry], answer: str) -> BuilderTurn: ...

    def opening(self, draft: ResumeDraft) -> BuilderTurn: ...


# ---------------------------------------------------------------- progress

def completeness(draft: ResumeDraft) -> list[tuple[str, bool]]:
    """The checklist on the builder page: each section and whether it has
    enough in it to print. Experience only counts once every role carries
    its dates, and a profile link is its own section: recruiters check."""
    return [
        ("Target role", bool(draft.headline.strip())),
        ("Name and contact", bool(draft.full_name.strip())
         and bool(draft.contact.email.strip() or draft.contact.phone.strip())),
        ("Experience", bool(draft.experience)
         and all(j.start.strip() and j.end.strip() for j in draft.experience)),
        ("Achievements", any(b.strip() for j in draft.experience for b in j.bullets)),
        ("Skills", any(s.strip() for s in draft.skills)),
        ("Education or certifications", any(e.strip() for e in draft.education + draft.certifications)),
        ("Summary", bool(draft.summary.strip())),
        ("Profile link", bool(draft.contact.links)),
    ]


def open_requirements(draft: ResumeDraft) -> list[str]:
    """The required follow-ups the draft itself still needs, in the order the
    interview asks them: dates on every role, then at least one profile link."""
    out = []
    if any(not j.start.strip() or not j.end.strip() for j in draft.experience):
        out.append("dates")
    if not draft.contact.links:
        out.append("links")
    return out


def missing_requirements(draft: ResumeDraft) -> list[str]:
    """The open requirements the interview should still ask for: those the
    person has not said they cannot give (the save step checks the draft
    itself either way)."""
    return [r for r in open_requirements(draft) if r not in draft.skipped]


# ---------------------------------------------------------------- rendering

def _clean(items: Sequence[str]) -> list[str]:
    return [" ".join(s.split()) for s in items if s.strip()]


def _section(title: str, lines: list[str]) -> list[str]:
    return [title.upper(), *lines] if lines else []


def render_resume_text(draft: ResumeDraft) -> str:
    """A plain-text resume: name, headline and contact on top, then summary,
    experience, skills, education, certifications and extras. Empty
    sections are left out and the layout is stable, so the same draft
    always renders the same text."""
    head = _clean([draft.full_name, draft.headline])
    contact = " | ".join(_clean([draft.contact.email, draft.contact.phone, draft.contact.location,
                                 *draft.contact.links]))
    if contact:
        head.append(contact)

    jobs: list[str] = []
    for job in draft.experience:
        title = " — ".join(_clean([job.title, ", ".join(_clean([job.company, job.location]))]))
        dates = " – ".join(_clean([job.start, job.end]))
        block = _clean([title, dates]) + [f"- {b}" for b in _clean(job.bullets)]
        if block:
            jobs.extend(([""] if jobs else []) + block)

    skills = _clean(draft.skills)
    sections = [
        _section("Summary", _clean([draft.summary])),
        _section("Experience", jobs),
        _section("Skills", [", ".join(skills)] if skills else []),
        _section("Education", [f"- {e}" for e in _clean(draft.education)]),
        _section("Certifications", [f"- {c}" for c in _clean(draft.certifications)]),
        _section("Additional", [f"- {x}" for x in _clean(draft.extras)]),
    ]
    blocks = ["\n".join(head)] if head else []
    blocks += ["\n".join(s) for s in sections if s]
    return "\n\n".join(blocks).strip()


# ---------------------------------------------------------------- scripted

_SKIP = re.compile(r"^(skip|pass|no|nope|none|nothing|n/?a|-+|no more|that'?s (it|all)|done|"
                   r"i (don'?t|do not) have (one|any)( yet)?|not applicable)$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")
_URL = re.compile(r"(https?://\S+|\b(www\.)?(linkedin|github|gitlab)\.com/\S+)", re.I)
_PHONE = re.compile(r"^\+?[\d\s().-]{7,}$")
MAX_JOBS = 8
# A paste this long is a whole old resume or profile, not one answer.
BLOCK_LINES = 12
BLOCK_CHARS = 900
# After a model's question the script can only guess the section, so a
# smaller block is already kept whole rather than misfiled.
GUESS_BLOCK_LINES = 3
GUESS_BLOCK_CHARS = 300


def is_skip(answer: str) -> bool:
    return bool(_SKIP.match(re.sub(r"[.!\s]+$", "", answer.strip().lower())))


def split_items(text: str) -> list[str]:
    """One item per line; a single line is split on commas or semicolons."""
    lines = [ln.strip(" \t-•*") for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) == 1:
        lines = [p.strip() for p in re.split(r"[;,]", lines[0]) if p.strip()]
    return lines


def _job_from(text: str) -> Job:
    # "Title, Company, Location, Start, End" (or one per line), positional
    # and verbatim: the script never guesses which part is which beyond order.
    parts = split_items(text) if "\n" in text.strip() else [p.strip() for p in text.split(",") if p.strip()]
    fields = ["title", "company", "location", "start", "end"]
    job = Job(**dict(zip(fields, parts)))
    if len(parts) > len(fields):
        job.end = ", ".join(parts[len(fields) - 1:])
    return job


def _contact_from(text: str, contact: Contact) -> Contact:
    contact = contact.model_copy(deep=True)
    for item in split_items(text):
        if _EMAIL.fullmatch(item):
            contact.email = item
        elif _URL.match(item) or re.match(r"^[\w.-]+\.(com|io|dev|net|org|me)(/\S*)?$", item):
            if item not in contact.links:
                contact.links.append(item)
        elif _PHONE.match(item):
            contact.phone = item
        else:
            contact.location = ", ".join(_clean([contact.location, item]))
    return contact


# (topic, question). "job"/"bullets" ask about the most recent job; the
# "more_*" pair repeats for each earlier job until the person says no.
SCRIPT: tuple[tuple[str, str], ...] = (
    ("role", "Let's build your resume. What job title are you aiming for next?"),
    ("name", "What is your full name, as it should appear at the top?"),
    ("contact", "How can employers reach you? Email, phone, city, and any links "
                "(LinkedIn, GitHub, portfolio) — separated by commas or one per line."),
    ("job", "Tell me about your most recent job: title, company, location, start and end dates, "
            "separated by commas (e.g. \"Data Analyst, Acme, Berlin, 2021, present\")."),
    ("bullets", "What did you achieve there? One result per line — numbers help "
                "(e.g. \"cut report time from 2 days to 2 hours\")."),
    ("more_job", "Any earlier job to add? Give its title, company, location and dates the same way, or say \"no\"."),
    ("more_bullets", "What did you achieve in that job? One result per line."),
    ("skills", "Which skills and tools should employers see? Separate them with commas."),
    ("education", "Your education: degree, school and year, one per line. Say \"skip\" if none."),
    ("certs", "Any certifications or licenses? One per line, or \"skip\"."),
    ("extras", "Anything else worth adding — languages, volunteering, awards? Or say \"no\"."),
    ("summary", "Last one: in two or three sentences, how would you sum yourself up for the role you want?"),
)
QUESTIONS = dict(SCRIPT)
_ORDER = [t for t, _ in SCRIPT]
DONE_QUESTION = ("That's everything I need. Save it with \"Save to my resumes\", then open Jobs — "
                 "your best matches are at the top. Apply to the roles you like and mark them "
                 "Applied; that is how a search turns into offers. I can help from the Assistant page too.")
LINKS_QUESTION = ("Employers check you up before they reply, so one real link makes this resume verifiable: "
                  "your LinkedIn, GitHub or a portfolio — which one should it show? (One is enough; if you "
                  "truly have none say \"skip\", but note the resume can only be saved with at least one.)")


def dates_question(draft: ResumeDraft) -> str:
    job = next(j for j in draft.experience if not j.start.strip() or not j.end.strip())
    who = " ".join((job.title or job.company or "that role").split())
    return (f"When did you start and end at {who}? Every role needs its dates — answer as: start, end "
            "(e.g. \"2021, present\").")


def _dates_from(text: str, job: Job) -> Job:
    parts = split_items(text) if "\n" in text else [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
    if not job.start:
        job.start = parts[0] if parts else job.start
        job.end = parts[1] if len(parts) > 1 else job.end
    elif parts:
        # The start is already known, so whatever they give now is the end
        # ("present", or a correction like "2019, 2022" -> the last part).
        job.end = parts[-1]
    return job


class ScriptedInterviewer:
    """The offline interviewer: fixed questions, answers filed verbatim."""

    def opening(self, draft: ResumeDraft) -> BuilderTurn:
        return BuilderTurn(draft=draft, question=QUESTIONS["role"], topic="role")

    @staticmethod
    def pending_topic(draft: ResumeDraft, history: Sequence[ChatEntry]) -> str:
        """The step the last question belongs to. After an LLM question (no
        topic) it is the first section the draft still lacks, so a fallback
        turn files the answer somewhere sensible."""
        last = next((e for e in reversed(history) if e.role == "assistant"), None)
        if last is not None and last.topic:
            return last.topic
        if last is not None and last.text == DONE_QUESTION:
            return "done"
        missing = {name for name, ok in completeness(draft) if not ok}
        for topic, section in (("role", "Target role"), ("name", "Name and contact"), ("job", "Experience"),
                               ("bullets", "Achievements"), ("skills", "Skills"),
                               ("education", "Education or certifications"), ("summary", "Summary")):
            if section == "Experience" and draft.experience:
                continue  # roles exist; what is missing is their dates, and a fallback must not add another job
            if section in missing:
                return "contact" if topic == "name" and draft.full_name.strip() else topic
        if "Profile link" in missing:
            return "contact"
        return "extras"

    async def next_turn(self, draft: ResumeDraft, history: Sequence[ChatEntry], answer: str) -> BuilderTurn:
        return self.file_answer(draft, self.pending_topic(draft, history), answer)

    def file_answer(self, draft: ResumeDraft, topic: str, answer: str, *, guessed: bool = False) -> BuilderTurn:
        """File ``answer`` under ``topic``. ``guessed`` means the question
        was the model's, so the topic is only a guess: then even a short
        multi-line answer is kept whole instead of split into a section it
        may not belong to."""
        draft = draft.model_copy(deep=True)
        text = answer.strip()
        note = ""
        lines = len(text.splitlines())
        if (len(text) >= BLOCK_CHARS or lines >= BLOCK_LINES
                or (guessed and (lines >= GUESS_BLOCK_LINES or len(text) >= GUESS_BLOCK_CHARS))):
            if topic not in ("bullets", "more_bullets", "summary"):
                # Without a model the script cannot sort a whole resume; keep
                # every word and pick out what is unambiguous.
                for email in _EMAIL.finditer(text):
                    draft.contact.email = draft.contact.email or email.group(0)
                for url in _URL.finditer(text):
                    if url.group(0) not in draft.contact.links:
                        draft.contact.links.append(url.group(0))
                draft.extras.append(text)
                if topic in ("dates", "links"):
                    question, done = (dates_question(draft) if topic == "dates" else LINKS_QUESTION), False
                else:
                    question, done = QUESTIONS.get(topic, DONE_QUESTION), topic == "done"
                return BuilderTurn(draft=draft, question=question, topic=topic, done=done, note=(
                                       "I couldn't sort that long answer, so I kept it whole under Additional "
                                       "(nothing is lost; edit it after saving) and picked out your email and "
                                       "links. Let's go on with the questions."))
        skip = is_skip(text)
        nxt = _ORDER[_ORDER.index(topic) + 1] if topic in _ORDER[:-1] else "done"
        if topic == "role" and not skip:
            draft.headline = text
        elif topic == "name" and not skip:
            draft.full_name = text
        elif topic == "contact" and not skip:
            draft.contact = _contact_from(text, draft.contact)
        elif topic in ("job", "more_job"):
            if skip:
                nxt = "skills"
            elif len(draft.experience) >= MAX_JOBS:
                note, nxt = f"That's {MAX_JOBS} jobs; older ones can wait for the saved resume.", "skills"
            else:
                draft.experience.append(_job_from(text))
                nxt = "bullets" if topic == "job" else "more_bullets"
        elif topic in ("bullets", "more_bullets"):
            if not skip and draft.experience:
                draft.experience[-1].bullets.extend(split_items(text) if "\n" in text else [text])
            elif not skip:
                draft.extras.append(text)
            nxt = "more_job"
        elif topic == "skills" and not skip:
            draft.skills.extend(s for s in split_items(text) if s not in draft.skills)
        elif topic == "education" and not skip:
            draft.education.extend(split_items(text) if "\n" in text else [text])
        elif topic == "certs" and not skip:
            draft.certifications.extend(split_items(text) if "\n" in text else [text])
        elif topic == "summary" and not skip:
            draft.summary = text
        elif topic == "links":
            if skip and "links" not in draft.skipped:
                draft.skipped.append("links")
            else:
                # Only fold it in when it plausibly contains a link; prose
                # ("I'd rather not") must not become a location.
                if _URL.search(text) or "@" in text:
                    draft.contact = _contact_from(text, draft.contact)
        elif topic == "dates":
            if skip and "dates" not in draft.skipped:
                draft.skipped.append("dates")
            elif draft.experience:
                first = next(j for j in draft.experience if not j.start.strip() or not j.end.strip())
                _dates_from(text, first)
        elif topic in ("extras", "done") and not skip:
            if topic == "done" and (open_reqs := open_requirements(draft)):
                # After the interview, typing first fills a missing requirement
                # (dates, a link) rather than landing in Additional — even one
                # previously skipped.
                return self.file_answer(draft, open_reqs[0], answer)
            draft.extras.extend(split_items(text) if "\n" in text else [text])
        if nxt == "done" or (nxt == "job" and draft.experience):
            # Never re-ask for a most-recent job when one is already on the
            # draft (a guessed topic can land here): open requirements come
            # first, then the flow continues after the job.
            fallback = "bullets" if nxt == "job" and draft.experience and not draft.experience[-1].bullets else "done"
            nxt = next(iter(missing_requirements(draft)), fallback)
        if nxt == "done":
            return BuilderTurn(draft=draft, question=DONE_QUESTION, done=True, note=note)
        if nxt == "dates":
            return BuilderTurn(draft=draft, question=dates_question(draft), topic="dates", note=note)
        if nxt == "links":
            return BuilderTurn(draft=draft, question=LINKS_QUESTION, topic="links", note=note)
        return BuilderTurn(draft=draft, question=QUESTIONS[nxt], topic=nxt, note=note)


# ---------------------------------------------------------------- LLM

INSTRUCTIONS = """\
You interview a job seeker to build their resume. Each turn you get the
current draft (JSON), the recent conversation and their newest answer. Return
the updated draft, your next question, and done=true only when every section
is covered or they asked to finish.

Rules:
- Merge the answer into the draft. Keep everything already there unless they
  correct it.
- NEVER invent facts. Use only what the person wrote. You may rephrase and
  tidy (e.g. turn "i made reports faster" into "Made reports faster"), but you
  must not add employers, job titles, dates, numbers, percentages, tools,
  skills, degrees or schools they did not state.
- Every role needs its start and end dates. When an answer adds a job without
  them, ask one short follow-up for just the dates before moving on; never
  invent or guess dates.
- Ask for at least one profile link - LinkedIn, GitHub or a portfolio - and
  say why: recruiters look the person up, and a real link makes the resume
  verifiable. One is enough; if they truly have none, note it and move on.
- Ask ONE short, focused question per turn, in this order: target role ->
  name and contact -> most recent job (title, company, dates), pressing once
  for measurable results -> earlier jobs -> skills -> education and
  certifications -> anything else. Draft the summary LAST, only from what was
  said, and ask them to confirm or edit it.
- If they paste a large block (an old resume or LinkedIn text), extract
  everything you can from it in this one turn and continue with what is
  still missing.
- If they say "skip", "no", or that they don't have one, move on.
- experience is most recent first. Bullets are short achievement lines.
- When done, your question steers the next step: save with the button on the
  right, then open Jobs - the best matches are at the top - and mark the ones
  they apply to as Applied. If they ask what to do next,
  give those concrete steps instead of a generic offer to help.
"""


class _LLMTurn(BaseModel):
    draft: ResumeDraft
    question: str = Field(description="The one next question to ask")
    done: bool = False


_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def invented_numbers(before: ResumeDraft, after: ResumeDraft, said: str) -> set[str]:
    """Numbers that appear in ``after`` but neither in ``before`` nor in
    anything the person said: the cheapest reliable sign of a made-up date
    or metric."""
    def nums(text: str) -> set[str]:
        return {n.replace(",", "") for n in _NUMBER.findall(text)}
    return nums(after.model_dump_json()) - nums(before.model_dump_json()) - nums(said)


EXTRACT_INSTRUCTIONS = """\
Turn a job seeker's own description of their background into a resume draft
(JSON). Rules:
- NEVER invent facts. Use only what they wrote. You may tidy their words into
  short achievement bullets, but must not add employers, job titles, dates,
  numbers, percentages, tools, skills, degrees or schools they did not state.
- experience is most recent first. Leave start or end empty when they gave no
  date for it; never guess one. A job they still hold ends "Present".
- headline is the role they want, else their current title.
- skills are the tools and skills they named, one per item, as they wrote them.
- education is one line per degree or course; languages, volunteering and
  awards go in extras. Leave summary empty.
"""

_WORDS = re.compile(r"[^a-z0-9+#]+")


def _words(text: str) -> str:
    return f" {_WORDS.sub(' ', text.lower()).strip()} "


def _said(value: str, text: str) -> bool:
    """``value`` appears, as whole words, in ``text`` (case and punctuation aside)."""
    v = _words(value)
    return bool(v.strip()) and v in _words(text)


def checked_extraction(draft: ResumeDraft, text: str) -> ResumeDraft:
    """A model's extraction held to what the person wrote: an invented
    number or employer rejects it (ValueError); a skill they never named is
    dropped. Returns the kept draft."""
    if made_up := invented_numbers(ResumeDraft(), draft, text):
        raise ValueError(f"model added numbers the person never gave: {sorted(made_up)[:5]}")
    if employers := [j.company for j in draft.experience if j.company and not _said(j.company, text)]:
        raise ValueError(f"model added employers the person never named: {employers[:3]}")
    seen: set[str] = set()
    skills = []
    for skill in draft.skills:
        key = _words(skill)
        if _said(skill, text) and key not in seen:
            seen.add(key)
            skills.append(skill)
    return draft.model_copy(update={"skills": skills})


class LLMInterviewer:
    """The interview run by a model. ``model`` is a Pydantic AI model or a
    zero-argument factory returning one (built lazily so a bad key fails a
    turn, not the startup). Any failure, timeout or invented number falls
    back to :class:`ScriptedInterviewer` for that turn."""

    # Free-tier models echoing a long draft took ~60 s in the live check.
    def __init__(self, model: object | Callable[[], object], *, timeout_s: float = 90.0,
                 history_turns: int = 16, model_id: str = "") -> None:
        self._model = model
        self.timeout_s = timeout_s
        self.history_turns = history_turns
        self.model_id = model_id
        self.script = ScriptedInterviewer()
        self._agent = None
        self._extract_agent = None

    def opening(self, draft: ResumeDraft) -> BuilderTurn:
        return self.script.opening(draft)

    def _get_agent(self):  # noqa: ANN202 - pydantic_ai types stay local
        if self._agent is None:
            from pydantic_ai import Agent, PromptedOutput

            model = self._resolve()
            # Prompted (JSON in text) output works on any chat model,
            # including free routers whose models lack tool calling.
            self._agent = Agent(model, output_type=PromptedOutput(_LLMTurn), instructions=INSTRUCTIONS,
                                retries=1)
        return self._agent

    def _resolve(self) -> object:
        return self._model() if callable(self._model) and not hasattr(self._model, "request") else self._model

    async def extract(self, text: str) -> ResumeDraft:
        """The whole draft from one background the person wrote (the
        Assistant's "make me a resume from this"). Unlike a turn there is no
        scripted fallback: a failure, timeout or invented fact raises, and the
        caller saves nothing."""
        if self._extract_agent is None:
            from pydantic_ai import Agent, PromptedOutput

            self._extract_agent = Agent(self._resolve(), output_type=PromptedOutput(ResumeDraft),
                                        instructions=EXTRACT_INSTRUCTIONS, retries=1)
        result = await asyncio.wait_for(self._extract_agent.run(text), self.timeout_s)
        return checked_extraction(result.output, text)

    def _prompt(self, draft: ResumeDraft, history: Sequence[ChatEntry], answer: str) -> str:
        recent = [{"role": e.role, "text": e.text[:4000]} for e in history[-self.history_turns:]]
        return (f"Current draft:\n{draft.model_dump_json()}\n\nRecent conversation:\n{json.dumps(recent)}\n\n"
                f"Newest answer:\n{answer}")

    async def next_turn(self, draft: ResumeDraft, history: Sequence[ChatEntry], answer: str) -> BuilderTurn:
        try:
            result = await asyncio.wait_for(self._get_agent().run(self._prompt(draft, history, answer)),
                                            self.timeout_s)
            out: _LLMTurn = result.output
            said = "\n".join([e.text for e in history if e.role == "user"] + [answer])
            if made_up := invented_numbers(draft, out.draft, said):
                raise ValueError(f"model added numbers the person never gave: {sorted(made_up)[:5]}")
            if not out.question.strip():
                raise ValueError("model returned no question")
            return BuilderTurn(draft=out.draft, question=out.question.strip(), done=out.done)
        except Exception as e:  # noqa: BLE001 - any model failure must keep the answer
            log.warning("resume builder LLM turn failed, using the script: %s: %s", type(e).__name__, e)
            last = next((h for h in reversed(history) if h.role == "assistant"), None)
            turn = self.script.file_answer(draft, self.script.pending_topic(draft, history), answer,
                                           guessed=last is None or not last.topic)
            reason = "took too long" if isinstance(e, TimeoutError) else "could not answer"
            turn.note = " ".join(_clean([f"The assistant {reason}, so I filed your answer as typed.", turn.note]))
            return turn


def build_resume_interviewer(settings: object) -> Interviewer:
    """The LLM interviewer when a chat provider (not TypeSafe's judges) has
    a key, else the scripted one."""
    from r3.agent.models import build_model
    from r3.agent.service import task_model_id

    available = [m.id for m in settings.available_models() if not m.id.startswith("typesafe:")]
    if not available:
        return ScriptedInterviewer()
    wanted = task_model_id(settings)
    model_id = wanted if wanted in available or wanted == settings.task_model else available[0]
    return LLMInterviewer(lambda: build_model(model_id, settings), model_id=model_id)
