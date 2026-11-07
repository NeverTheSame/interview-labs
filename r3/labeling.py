"""The job-labeling rubric: the questions the operator answers about one
posting on ``/label``, and the parser that turns a submitted form into the
``answers`` JSON stored in ``r3_job_labels``.

The rubric is versioned (:data:`RUBRIC_VERSION`). A label row carries the
version it was made under; changing a question, an option or its meaning
means a new version, never an edit of this one, so old labels stay readable.
Every option has a one-sentence description: the page shows it as the
button's tooltip, and the alignment eval (``evals/alignment.py``) hands the
same sentences to Jev as the option criteria, so the operator and the model
are asked the same question in the same words.

Answers are stored per question key. A question the operator did not answer
is stored as :data:`SKIPPED` (the normal case after ``relevant = no``); the
eval excludes ``skipped`` and each question's "can't tell" option from the
agreement numbers and reports them apart.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

RUBRIC_VERSION = "job-label-rubric/v1"
SKIPPED = "skipped"
DEFAULT_LABELER = "operator"
NOTE_MAX_CHARS = 1000
SECONDS_MAX = 24 * 3600

Kind = Literal["score", "choice", "multi"]


@dataclass(frozen=True)
class Option:
    value: str
    label: str
    description: str


@dataclass(frozen=True)
class Question:
    key: str
    title: str
    kind: Kind
    options: tuple[Option, ...]
    unclear: str  # the value that means "can't tell from this text"
    hint: str = ""  # the judging instruction shown under the title
    exclusive: str = ""  # multi only: the option that stands alone ("none")

    @property
    def values(self) -> tuple[str, ...]:
        return tuple(o.value for o in self.options)

    @property
    def keys(self) -> tuple[str, ...]:
        """The key that picks each option, in option order: 1 to 9, then 0."""
        return OPTION_KEYS[:len(self.options)]

    @property
    def required(self) -> bool:
        return self.kind == "score"

    def option(self, value: str) -> Option | None:
        return next((o for o in self.options if o.value == value), None)


RELEVANT = Question(
    "relevant", "Would I apply to this?", "score",
    (
        Option("yes", "Yes", "I would apply: an SRE, DevOps, platform, data or AI engineering role I could do from "
                             "Canada, remote or clearly open to it, at a level that fits me."),
        Option("maybe", "Maybe", "Worth a closer look: it mostly fits, but something is unclear or off (the level, the "
                                 "location wording, the contract terms, whether Canada is fine)."),
        Option("no", "No", "I would not apply: the wrong kind of role, not open to someone based in Canada, on-site "
                           "somewhere else, too junior or too senior, or a red flag."),
    ),
    unclear="maybe",
)

ROLE_BUCKET = Question(
    "role_bucket", "What is this job really?", "choice",
    (
        Option("sre", "SRE", "Site reliability engineering: on-call, SLOs, incident response, the reliability of "
                             "production systems."),
        Option("devops", "DevOps", "DevOps: CI/CD, build and release, infrastructure automation, developer tooling."),
        Option("platform_cloud", "Platform / cloud", "Platform or cloud engineering: internal platforms, Kubernetes, "
                                                     "AWS/GCP/Azure infrastructure, infrastructure as code."),
        Option("data_eng", "Data eng", "Data engineering: pipelines, warehouses, ETL/ELT, data platforms."),
        Option("ml", "ML eng", "ML engineering: training, evaluating and serving models, feature and training "
                               "pipelines, MLOps, ML platforms."),
        Option("ai", "AI eng", "AI engineering: building products on LLMs and other foundation models (agents, "
                               "RAG, prompt and eval tooling, fine-tuning for a product) rather than training "
                               "models."),
        Option("backend", "Backend", "Backend software engineering: services and APIs without an infrastructure "
                                     "focus."),
        Option("other_eng", "Other eng", "Another engineering role: frontend, mobile, QA, security, embedded, "
                                         "hardware."),
        Option("not_eng", "Not eng", "Not an engineering role: sales, recruiting, design, support, or management "
                                     "with no hands-on work."),
        Option("unclear", "Unclear", "The text does not say what the job really is."),
    ),
    unclear="unclear",
)

WORK_MODE = Question(
    "work_mode", "Work mode", "choice",
    (
        Option("remote", "Remote", "Remote: the work is done from home or anywhere; occasional travel or a rare "
                                   "on-site visit still counts as remote."),
        Option("hybrid", "Hybrid", "Hybrid: a set number of days a week or month in an office."),
        Option("onsite", "On-site", "On-site: in an office or facility every working day."),
        Option("unclear", "Unclear", "The description does not say; the location field alone does not count."),
    ),
    unclear="unclear",
    hint="Judge from the description, not the location field.",
)

EMPLOYMENT = Question(
    "employment", "Employment type", "choice",
    (
        Option("fulltime", "Full-time", "Full-time permanent employment, stated as such."),
        Option("contract", "Contract", "Contract, fixed-term or freelance work, stated as such (W-2, T4 or through "
                                       "an agency)."),
        Option("c2c", "C2C", "Corp-to-corp: the posting says C2C, 1099 or incorporated contractors only."),
        Option("parttime", "Part-time", "Part-time hours, stated as such."),
        Option("internship", "Internship", "An internship, co-op or student placement."),
        Option("unstated", "Unstated", "The text does not state the employment type; a yearly salary or benefits "
                                       "alone do not count."),
    ),
    unclear="unstated",
    hint="Only what the text states. If it only implies it (salary, benefits), pick unstated.",
)

ELIGIBLE_FROM_CANADA = Question(
    "eligible_from_canada", "Eligible from Canada?", "choice",
    (
        Option("yes", "Yes", "Nothing stops an applicant based in Canada: remote across North America or worldwide, "
                             "or Canada is named."),
        Option("no", "No", "Something does: US citizenship or a security clearance, US only or specific US states, "
                           "attendance at an office in another country, or no work permit sponsorship where one "
                           "would be needed."),
        Option("unclear", "Unclear", "The text does not say either way."),
    ),
    unclear="unclear",
    hint="Citizenship, clearance, \"US only\", office attendance.",
)

SENIORITY_FIT = Question(
    "seniority_fit", "Seniority", "choice",
    (
        Option("too_junior", "Too junior", "Entry-level or junior: below a senior engineer with years of production "
                                           "experience."),
        Option("fits", "Fits", "Mid-level to senior or staff individual contributor, or a small hands-on team lead."),
        Option("too_senior", "Too senior", "Director, VP, head of, or a principal role that is mostly management or "
                                           "strategy."),
        Option("unclear", "Unclear", "The text does not say what level it wants."),
    ),
    unclear="unclear",
)

RED_FLAG = Question(
    "red_flag", "Red flags", "multi",
    (
        Option("clearance", "Clearance", "Requires a security clearance, or eligibility for one."),
        Option("talent_pool_evergreen", "Talent pool", "Not a real opening: a talent pool, an evergreen requisition, "
                                                       "\"future opportunities\"."),
        Option("ai_trainer_gig", "AI trainer gig", "An AI data-labeling or model-training gig: paid per task, \"AI "
                                                   "trainer\", \"expert annotator\", \"rate your conversations\"."),
        Option("staffing_spam", "Staffing spam", "A staffing agency posting with no identifiable client, duplicated "
                                                 "across boards, or too vague to be a real job."),
        Option("relocation_required", "Relocation", "Requires relocating to take the job."),
        Option("none", "None", "None of these."),
    ),
    unclear="none",
    exclusive="none",
)

QUESTIONS: tuple[Question, ...] = (RELEVANT, ROLE_BUCKET, WORK_MODE, EMPLOYMENT, ELIGIBLE_FROM_CANADA, SENIORITY_FIT,
                                   RED_FLAG)
BY_KEY: dict[str, Question] = {q.key: q for q in QUESTIONS}

# A form picks an option by its position with one key: 1 to 9, then 0 for a
# tenth. No question may have more options than that.
OPTION_KEYS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
assert all(len(q.options) <= len(OPTION_KEYS) for q in QUESTIONS)


@dataclass(frozen=True)
class Rubric:
    version: str
    questions: tuple[Question, ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(q.key for q in self.questions)


RUBRIC = Rubric(RUBRIC_VERSION, QUESTIONS)

_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def parse_answers(values: Mapping[str, Sequence[str]], rubric: Rubric = RUBRIC) -> dict[str, Any]:
    """The ``answers`` JSON for a submitted form (field name -> the values
    sent, as ``request.form().getlist`` gives them). ``relevant`` is
    required; every other question the form leaves empty is stored as
    :data:`SKIPPED`. A multi-select is stored as a list in option order;
    its exclusive option ("none") may not be combined with another. Raises
    ``ValueError`` with a message the page can show."""
    answers: dict[str, Any] = {}
    for q in rubric.questions:
        sent = [v.strip() for v in values.get(q.key, ()) if isinstance(v, str) and v.strip()]
        if not sent:
            if q.required:
                raise ValueError(f"{q.key} is required")
            answers[q.key] = SKIPPED
            continue
        bad = [v for v in sent if v not in q.values]
        if bad:
            raise ValueError(f"{q.key}: {bad[0]!r} is not one of {', '.join(q.values)}")
        if q.kind == "multi":
            chosen = [v for v in q.values if v in sent]
            if q.exclusive and q.exclusive in chosen and len(chosen) > 1:
                raise ValueError(f"{q.key}: {q.exclusive!r} cannot be combined with another option")
            answers[q.key] = chosen
        else:
            if len(set(sent)) > 1:
                raise ValueError(f"{q.key}: pick one option, not {len(set(sent))}")
            answers[q.key] = sent[0]
    return answers


def clean_note(raw: str | None) -> str:
    """One line, control characters dropped, at most :data:`NOTE_MAX_CHARS`."""
    text = _CONTROL.sub(" ", (raw or "").replace("\r", " ").replace("\n", " "))
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) > NOTE_MAX_CHARS:
        raise ValueError(f"note is too long (at most {NOTE_MAX_CHARS} characters)")
    return text


def clean_session_id(raw: str | None) -> str:
    raw = (raw or "").strip()
    return raw if _SESSION_ID.match(raw) else ""


def clean_seconds(raw: str | None) -> int:
    try:
        n = int(float((raw or "").strip()))
    except ValueError:
        return 0
    return min(max(n, 0), SECONDS_MAX)


def answered(answers: Mapping[str, Any], key: str) -> bool:
    """Whether a stored answer is a real one: neither skipped nor the
    question's "can't tell" option."""
    q = BY_KEY.get(key)
    value = answers.get(key)
    if q is None or value == SKIPPED or value is None:
        return False
    if q.kind == "multi":
        return isinstance(value, list) and bool(value) and value != [q.unclear]
    return value != q.unclear


def summarize(answers: Mapping[str, Any]) -> str:
    """One line for a flash message or a history row: ``relevant: yes · sre · remote``."""
    parts: list[str] = []
    for q in QUESTIONS:
        value = answers.get(q.key)
        if value is None or value == SKIPPED:
            continue
        shown = ", ".join(value) if isinstance(value, list) else str(value)
        parts.append(f"{q.key}: {shown}" if q is RELEVANT else shown)
    return " · ".join(parts)


__all__ = [
    "BY_KEY", "DEFAULT_LABELER", "ELIGIBLE_FROM_CANADA", "EMPLOYMENT", "NOTE_MAX_CHARS", "OPTION_KEYS", "QUESTIONS",
    "RED_FLAG",
    "RELEVANT", "ROLE_BUCKET", "RUBRIC", "RUBRIC_VERSION", "SENIORITY_FIT", "SKIPPED", "WORK_MODE", "Option",
    "Question", "Rubric", "answered", "clean_note", "clean_seconds", "clean_session_id", "parse_answers", "summarize",
]
