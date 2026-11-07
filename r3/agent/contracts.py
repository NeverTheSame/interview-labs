"""Contracts between the router, the advisor, the web layer, the panel and
the evals. A user message becomes a :class:`RouteDecision`; the decision
becomes an :class:`Action` the right-hand panel renders; the advisor narrates
it on the left. Nothing in here calls a model or a database."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from r3.db import Job
from r3.runs.models import StatsSnapshot, WorkflowKind, WorkflowRun

Intent = Literal["advice", "jobs", "stats"]
INTENTS: tuple[Intent, ...] = ("advice", "jobs", "stats")

INTENT_DESCRIPTIONS: dict[str, str] = {
    "advice": "A question, opinion or plan about the job search: answer in conversation, no workflow runs.",
    "jobs": "Show, list, filter or count the jobs already stored, without changing anything.",
    "stats": "Statistics, metrics, numbers, progress, a dashboard, a summary of the search's state, or how the runs went.",
}

# Intents that start a workflow, and the workflow they start.
WORKFLOW_INTENTS: dict[str, WorkflowKind] = {"stats": "stats"}


class RouteParams(BaseModel):
    """What the message says about the work, all optional. Free text is
    normalised later against the database's real values (role categories,
    tech tags)."""

    model_config = ConfigDict(extra="forbid")
    roles: list[str] = Field(default_factory=list, max_length=10)
    remote: bool | None = None
    location: str = Field(default="", max_length=120)
    tech: list[str] = Field(default_factory=list, max_length=10)
    employment: str = Field(default="", max_length=40)  # fulltime | contract | c2c | ...
    min_salary: int | None = Field(default=None, ge=0, le=10_000_000)
    days: int | None = Field(default=None, ge=1, le=365)
    limit: int | None = Field(default=None, ge=1, le=50)
    job_ids: list[int] = Field(default_factory=list, max_length=25)
    search: str = Field(default="", max_length=200)
    # A job status the message names ("show applied jobs"): new | applied |
    # skipped | expired, or "all" for every status. Empty means the action's default.
    # Spellings like "unreviewed" are normalised; anything else becomes "".
    status: str = Field(default="", max_length=40)

    @field_validator("roles", "tech")
    @classmethod
    def _strip(cls, values: list[str]) -> list[str]:
        return [v.strip()[:80] for v in values if v and v.strip()]

    @field_validator("status")
    @classmethod
    def _status(cls, value: str) -> str:
        return _STATUS_ALIASES.get(value.strip().lower().replace("-", "_").replace(" ", "_"), "")


_STATUS_ALIASES: dict[str, str] = {
    "new": "new", "unreviewed": "new", "not_reviewed": "new", "untriaged": "new",
    "applied": "applied", "skipped": "skipped", "expired": "expired", "all": "all", "any": "all", "every": "all",
}


class RouteDecision(BaseModel):
    """The router's verdict for one message."""

    intent: Intent
    params: RouteParams = Field(default_factory=RouteParams)
    confidence: float | None = Field(default=None, ge=0, le=1)  # reported by Jev; None otherwise
    router: Literal["rules", "llm", "jev", "fallback"] = "llm"
    reason: str = Field(default="", max_length=300)

    @property
    def workflow(self) -> WorkflowKind | None:
        return WORKFLOW_INTENTS.get(self.intent)


# --- chat turn -------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=16000)

    @field_validator("content")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be blank")
        return value


class ChatTurnRequest(BaseModel):
    """One turn: the conversation so far plus the new user message, which is
    the last entry. ``mode`` = "advice" skips workflow detection."""

    model_config = ConfigDict(extra="forbid")
    messages: list[ChatMessage] = Field(min_length=1, max_length=41)
    model: str | None = Field(default=None, max_length=160)
    include_context: bool = True
    mode: Literal["auto", "advice"] = "auto"

    @model_validator(mode="after")
    def _valid_conversation(self) -> Self:
        if self.messages[-1].role != "user":
            raise ValueError("Conversation must end with a user message")
        for i, message in enumerate(self.messages):
            if message.role != ("user" if i % 2 == 0 else "assistant"):
                raise ValueError("Messages must alternate between user and assistant")
            if message.role == "user" and len(message.content) > 8000:
                raise ValueError("User messages must be at most 8000 characters")
        if sum(len(m.content) for m in self.messages) > 80000:
            raise ValueError("Conversation is too long; start a new chat")
        return self

    @property
    def text(self) -> str:
        return self.messages[-1].content

    @property
    def history(self) -> list[ChatMessage]:
        return self.messages[:-1]


# --- panel actions ----------------------------------------------------------------


class JobCard(BaseModel):
    """A job as the panel lists it. A regular account's card carries no
    posting link and no source board (the OQ-6 teaser): ``full`` is for
    admins and the local operator only."""

    id: int
    position: str
    company: str = ""
    location: str = ""
    status: str = ""
    salary: str = ""
    site_name: str = ""
    job_url: str = ""
    role_category: str = ""
    scraped_at: datetime | None = None

    @classmethod
    def from_job(cls, job: Job, *, full: bool) -> "JobCard":
        return cls(id=job.id, position=job.position, company=job.company, location=job.location,
                   status=job.status, salary=job.salary, site_name=job.site_name if full else "",
                   job_url=job.job_url if full else "", role_category=job.role_category, scraped_at=job.scraped_at)


class AdviceAction(BaseModel):
    """Nothing for the panel: the advisor answers."""

    kind: Literal["advice"] = "advice"


class JobsAction(BaseModel):
    """A job list for the panel, with the /jobs query string that reproduces it."""

    kind: Literal["jobs"] = "jobs"
    title: str
    filter_query: str = ""  # e.g. "?status=new&role=SRE&remote=1"
    jobs: list[JobCard] = Field(default_factory=list)
    total: int = 0


class StatsAction(BaseModel):
    kind: Literal["stats"] = "stats"
    snapshot: StatsSnapshot
    run: WorkflowRun | None = None  # when the snapshot was recorded as a run


class UnavailableAction(BaseModel):
    """A workflow was asked for but cannot run right now (the database did not answer)."""

    kind: Literal["unavailable"] = "unavailable"
    intent: Intent
    reason: str
    hint: str = ""


class ResumeFromTextAction(BaseModel):
    """Make a resume from what the person wrote: the page posts ``text`` to
    ``POST /api/resume/from-text`` and shows what R3 read, like an upload."""

    kind: Literal["resume_text"] = "resume_text"
    text: str


Action = Annotated[
    Union[AdviceAction, JobsAction, StatsAction, UnavailableAction,
          ResumeFromTextAction],
    Field(discriminator="kind"),
]
