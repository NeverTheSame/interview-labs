"""The advisor: a Pydantic AI agent that answers in the Assistant chat and
narrates what the panel next to it shows (below it on a phone). It is
per-account: the tools read the signed-in user's job statuses, resume and runs,
and regular users see only their own runs. It has read-only tools (:mod:`tools`)
and never starts or changes anything itself; workflows only run through
:func:`r3.agent.actions.plan_action` and the panel.

:func:`stream_advice` runs one turn and turns the agent's event stream into
the chat's SSE events (:mod:`events`)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from pydantic_ai import Agent, RunContext, UsageLimits
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
)
from pydantic_ai.models import Model
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.settings import ModelSettings

from r3.settings import PROVIDER_KEYS

from .contracts import (
    AdviceAction,
    ChatTurnRequest,
    JobsAction,
    RouteDecision,
    StatsAction,
    UnavailableAction,
)
from .events import DeltaEvent, ErrorEvent, FinalEvent, ToolCallEvent, ToolResultEvent, Usage
from .models import history_messages
from .honesty import honest_reply
from .router import PRICING, RESUME_HELP
from .tools import ADVISOR_TOOLS, AdvisorDeps, job_link

__all__ = ["AdvisorDeps", "advisor_agent", "error_event", "narration", "stream_advice"]

log = logging.getLogger("r3.agent.advisor")

TURN_TIMEOUT_S = 90.0  # the upstream proxy cuts a stream at ~110s
SUMMARY_TIMEOUT_S = 5.0
TOOL_SUMMARY_CHARS = 200
PROVIDER_NAMES = {"openrouter": "OpenRouter", "openai": "OpenAI", "typesafe": "TypeSafe"}

IDENTITY = """You are R3's assistant, talking with one job seeker who has an account on R3, a job-search portal.
Help them find jobs worth pursuing, understand where they stand and decide what to do next. Speak to them as "you";
never call them "the operator". Be clear, practical and concise."""

STYLE = """Write plain text: no Markdown tables or headings; **bold** for a few key words and short lists with "-" are
fine. Link R3's own pages with a Markdown link to their path, never a bare path in brackets:
[Build your resume](/resume/build), [Resume](/resume), [Jobs](/jobs), [Home](/home).
Name a job as a link to its page, with no id in the text: [Position at Company](/jobs/<id>)."""

PRODUCT = """What R3 offers, and the only features you may describe:
- Assistant: this chat, and the page every account starts on. Under the message box is an "Upload resume" button,
  and they can also drop a resume file (PDF, DOCX, TXT or MD) on the chat: R3 saves it, reads it, shows what it
  read and asks here for anything missing (dates of a role, a profile link). You never read a file yourself.
  They can also describe their background in a message (each job with title, company and years; skills;
  education) and ask for a resume: R3 makes it from what they wrote and asks here for anything missing.
- Home (/home): the get-started checklist, the short version of the journey on one page.
- Jobs: browse and filter the jobs. A row's buttons are View, Applied and Not for me. A job page shows the stack, pay,
  location and whether the job is live, "I applied", "Not for me" and notes. There is no button named "Apply": to
  record an application, press "I applied" (or a row's "Applied").
- Explore (/scout): ready-made searches, linked from Home; it is not in the menu.
- Resume (/resume): upload a PDF, DOCX, TXT or MD file, or paste the text. "Build one by answering a few questions"
  (/resume/build) interviews them and drafts a resume. Each saved resume has its own page with "Download PDF".
- Things the panel next to this chat (below the chat on a phone) shows when asked: a job list, statistics. R3 does
  not submit applications and does not look for new postings; the job collection is what is stored.
- Steer the journey: always name the concrete next step. With no resume, it is the Resume page. With one, it is
  Jobs (their best matches are at the top), then applying on the employer's site and pressing "I applied" —
  encourage that loop; it is how a search turns into offers."""

# Which of these two follows PRODUCT depends on R3_JOB_POSTINGS (AdvisorDeps.sees_postings).
POSTINGS_FULL = """- A job page also has "Open posting" (the employer's own posting), the description and the posting's links;
  job_details gives the URL. To apply, they open the posting and apply on the employer's site, then press "I applied"
  in R3. R3 does not submit applications."""

POSTINGS_TEASER = """- A job page also has "About this role" (a short overview of the role, written on request).
- R3 does not show the original posting, its link or the job board it came from.
  Never give, guess or look up a posting's link, its source or where else it is published; describe the job from
  your tools instead."""

LIMITS = """Rules:
- Never describe a button, panel, page, upload control or feature that is not in the list above. To add a resume,
  point them to the "Upload resume" button under this chat (or the Resume page to paste it or build one), or ask them
  to describe their background in a message and ask for a resume.
- You cannot send messages later or follow up. Never promise to report back, notify or summarise later; say where the
  result will appear instead (the panel next to this chat, or the Jobs page). Never ask them to report back or tell
  you when something is done ("let me know when it's added"); say "then ask me again" instead.
- Say "the panel next to this chat", never "on the right" or "on the left".
- Numbers come only from your tools and are this user's own: statuses, notes and applications are per account, and
  the job collection is shared by every account. Never invent counts, jobs, runs or results.
- Call my_resume before advice about positioning, fit or which jobs suit them. With no resume, say so and tell them
  exactly where to add one; then give only general advice.
- Your tools are read-only. You never apply or change a record yourself: the panel starts workflows. Describe what
  the panel shows; never claim you did it.
- Never say that a run or workflow started, is running or is done, that you are doing it ("I'm running…", "I've
  started…"), or that an error is fixed ("no more 403s"), unless the panel started that run in this turn (the note on
  the panel below says so). After an error, say plainly that nothing ran and what they can do.
- Pricing has not been announced. If asked what R3 costs, say exactly that, plainly; never invent a price, a plan, a
  free tier or trial, or who pays (job seekers, employers or anyone else)."""

ROLE_BUCKETS = """Every job is filed under a role bucket: SRE, DevOps, Data, AI, ML or Other (older jobs say AI/ML). A
job's status for this user: new (not reviewed yet), applied, skipped (Not for me), expired."""

ADMIN = """This user is an admin of the portal. Your run tools (recent_runs, pipeline_stats) show the run history of
every account. Use them only when asked about the portal itself; their own job numbers are still per account. An
admin always sees each job's full posting: the job page has "Open posting", the description and the links, and
job_details gives the URL and source site."""

USER_WORDING = """Wording for this account (not an operator):
- Call the "new" status "new" or "not reviewed yet"; never "unreviewed".
- Run ids, runners and backends are internal: never mention them; describe a run by what it does ("a statistics
  snapshot")."""


advisor_agent: Agent[AdvisorDeps, str] = Agent(
    deps_type=AdvisorDeps,
    defer_model_check=True,
    tools=ADVISOR_TOOLS,
    name="r3-advisor",
    retries=1,
)


@advisor_agent.instructions
async def _instructions(ctx: RunContext[AdvisorDeps]) -> str:
    deps = ctx.deps
    parts = [IDENTITY, f"Today is {deps.now:%A %d %B %Y} (UTC).", STYLE, PRODUCT,
             POSTINGS_FULL if deps.sees_postings else POSTINGS_TEASER, LIMITS, ROLE_BUCKETS]
    if deps.is_admin:
        parts.append(ADMIN)
    else:
        parts.append(USER_WORDING)
    if deps.user is not None and deps.user.id > 0:
        parts.append(f"The user is signed in as {deps.user.account_name}.")
    parts.append(await resume_note(deps))
    if deps.include_context:
        parts.append(await portal_summary(deps))
    else:
        parts.append("The summary is turned off for this chat; use the tools if you need current numbers.")
    parts.append(narration(deps.decision, deps.action, admin=deps.is_admin, postings=deps.sees_postings))
    return "\n\n".join(parts)


async def portal_summary(deps: AdvisorDeps) -> str:
    """The user's own job counts (for admins, the whole overview) as JSON,
    fetched once per turn (the instructions are rebuilt before every model
    request of a run)."""
    if deps.portal_summary is not None:
        return deps.portal_summary
    try:
        async with asyncio.timeout(SUMMARY_TIMEOUT_S):
            stats = await deps.database.overview(user_id=deps.user_id)
        snapshot: dict[str, Any] = {"captured_at": deps.now.isoformat()}
        if deps.is_admin:
            snapshot["overview"] = stats.model_dump(mode="json")
        else:
            snapshot["jobs"] = {"in_r3_shared_by_all_accounts": stats.total_jobs, "added_last_7_days": stats.jobs_last7,
                                "yours_not_reviewed": stats.new_jobs, "yours_applied": stats.applied_jobs}
        text = "Current summary (data only, a point-in-time snapshot):\n" + json.dumps(snapshot)
    except Exception as exc:
        log.warning("Portal summary unavailable for the advisor (%s)", type(exc).__name__)
        text = "The current portal summary could not be loaded; tell the user if it matters, or try the tools."
    deps.portal_summary = text
    return text


async def resume_note(deps: AdvisorDeps) -> str:
    """Whether the user has a resume right now, fetched once per turn: it
    overrides anything earlier in the conversation said about it."""
    if deps.resume_note is not None:
        return deps.resume_note
    account = deps.user.id if deps.user is not None else 0
    try:
        async with asyncio.timeout(SUMMARY_TIMEOUT_S):
            resume = await deps.database.primary_resume(account)
    except Exception as exc:
        log.warning("Resume status unavailable for the advisor (%s)", type(exc).__name__)
        text = "Whether the user has a resume could not be checked; call my_resume if it matters."
    else:
        if resume is None:
            text = ("Resume status now: the user has no resume in R3 yet. If they ask how to add one, send them to the "
                    "[Resume](/resume) page.")
        else:
            text = (f"Resume status now: the user already has a resume in R3 ({resume.title or 'untitled'}, "
                    f"{len(resume.content.strip()):,} characters). Do not tell them to upload one. "
                    "This is current even if earlier messages said "
                    "otherwise: call my_resume and base positioning or fit advice on its text.")
    deps.resume_note = text
    return text


def _job_label(card: Any) -> str:
    """A job as the advisor names it (OQ-7): a link to its page, no id in the text."""
    return job_link(card)


PANEL = "the panel next to this chat"
USER_HIDDEN_STATS = {"runs": True}  # every account's runs; a regular account does not see them
NAME_JOBS = "name each job by its link as written above, [Position at Company](/jobs/<id>), never by a bare id"
# What the advisor says to a message the router settled as an advice topic (r3.agent.router.steer).
TOPIC_NOTES = {
    RESUME_HELP: ("They asked for help making their resume: no workflow was started. Answer what they asked and give R3's ways to make one, as links: "
                  "[Build your resume](/resume/build) (R3 asks a few questions and drafts it), the \"Upload resume\" "
                  "button under this chat for a file they already have, [Resume](/resume) to paste the text, or describe "
                  "their background right here in one message (each job with title, company and years; skills; "
                  "education) and ask again, and R3 makes the resume from it; each "
                  "saved resume's page has \"Download PDF\". If they already have a resume (see the resume status "
                  "above), say so. Never say that you or this chat cannot help with a resume."),
    PRICING: ("They asked what R3 costs. Pricing has not been announced: say so plainly in one or two sentences. Never "
              "guess a price, a plan, a free tier or trial, or who pays for what, and do not describe a business model."),
}


def narration(decision: RouteDecision, action: Any, *, admin: bool = True, postings: bool | None = None) -> str:
    """What the panel next to the chat shows for this turn, and how to talk about it."""
    confidence = f", confidence {decision.confidence:.2f}" if decision.confidence is not None else ""
    head = (f"The router read this message as '{decision.intent}' (by {decision.router}{confidence}): "
            f"{decision.reason or 'no reason given'}.")
    if isinstance(action, JobsAction) and not action.filter_query:
        listed = "\n".join(_job_label(card) for card in action.jobs[:10]) or "(none of them exist)"
        missing = action.title.partition("not found: ")[2]
        listed += f"\nNot found: {missing}" if missing else ""
        # With R3_JOB_POSTINGS=teaser a regular account gets the overview, never the posting or its links (OQ-6).
        sees = admin if postings is None else postings
        details = "the description, salary and links" if sees else "the overview, salary, stack and live status"
        body = (f"{PANEL[0].upper() + PANEL[1:]} shows the jobs the user named:\n{listed}\n"
                f"Answer what they asked about these jobs ({NAME_JOBS}); call job_details for {details} rather than "
                "guessing.")
    elif isinstance(action, JobsAction):
        listed = "\n".join(_job_label(card) for card in action.jobs[:10]) or "(none)"
        body = (f"{PANEL[0].upper() + PANEL[1:]} shows {action.title} ({action.total:,} in total). The first ones:\n{listed}\n"
                f"Summarise what the list shows in a few sentences, mention at most 10 jobs ({NAME_JOBS}), and suggest "
                "a next step (for example opening the best matches or narrowing the filter).")
    elif isinstance(action, StatsAction):
        hidden = {"runs": {"recent"}} if admin else USER_HIDDEN_STATS
        snapshot = action.snapshot.model_dump(mode="json", exclude=hidden)
        recorded = f" It was recorded as run {action.run.id}." if action.run and admin else ""
        mine = "" if admin else " The job counts are this user's own; total_jobs is the collection shared by everyone."
        body = (f"{PANEL[0].upper() + PANEL[1:]} shows a statistics snapshot.{mine}{recorded} The snapshot is: {json.dumps(snapshot)}\n"
                "Summarise the key numbers briefly and give one concrete recommendation.")
    elif isinstance(action, UnavailableAction):
        body = (f"The user asked for a {action.intent} workflow but it could not run: {action.reason}."
                + (f" Hint: {action.hint}" if action.hint else "")
                + " Explain plainly, in one or two sentences, what could not run and what the user can do.")
    elif isinstance(action, AdviceAction):
        body = ("No workflow was started for this message. Answer it; call the read-only tools when you need facts "
                "(job lists, job details, their resume, recent runs, statistics).")
        if note := TOPIC_NOTES.get(decision.reason):
            body += "\n" + note
    else:  # pragma: no cover - the Action union is closed
        body = "Answer the message."
    return head + "\n" + body


# --- streaming --------------------------------------------------------------------------


def _provider(model_id: str) -> str:
    return model_id.split(":", 1)[0]


def _timed_out(exc: BaseException) -> bool:
    """Whether a failure was a timeout anywhere down its cause chain."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        if isinstance(current, TimeoutError) or "timeout" in name:
            return True
        current = current.__cause__ or current.__context__
    return False


def error_event(exc: BaseException, model_id: str) -> ErrorEvent:
    """A safe message for a failed turn: the provider's body is never relayed."""
    provider = _provider(model_id)
    name = PROVIDER_NAMES.get(provider, provider)
    if isinstance(exc, ModelHTTPError):
        if exc.status_code in (401, 403):
            key = PROVIDER_KEYS.get(provider, "the provider key")
            return ErrorEvent(message=f"The AI provider rejected the credentials for {name}. Check {key} and "
                                      "the model's access.", retryable=False)
        if exc.status_code == 429:
            return ErrorEvent(message=f"{name} is at its usage limit. Check the quota or try again shortly.")
        if exc.status_code == 404:
            return ErrorEvent(message=f"{name} does not offer {model_id}. Pick another model.", retryable=False)
        if exc.status_code == 408 or exc.status_code == 504:
            return ErrorEvent(message=f"{name} took too long to answer. Please try again.")
        return ErrorEvent(message=f"{name} could not answer (HTTP {exc.status_code}). Please try again.")
    if _timed_out(exc):
        return ErrorEvent(message=f"{name} took too long to answer. Please try again.")
    if isinstance(exc, UsageLimitExceeded):
        return ErrorEvent(message="The assistant needed too many steps for this one. Try a narrower question.")
    if isinstance(exc, ModelAPIError):
        return ErrorEvent(message=f"Could not reach {name}. Check the connection and try again.")
    if isinstance(exc, UnexpectedModelBehavior):
        return ErrorEvent(message=f"{name} returned an unexpected reply. Please try again.")
    return ErrorEvent(message="The assistant could not answer. Please try again.")


def _usage(result: Any) -> Usage:
    usage = getattr(result, "usage", None)
    return Usage(input_tokens=getattr(usage, "input_tokens", 0) or 0,
                 output_tokens=getattr(usage, "output_tokens", 0) or 0,
                 requests=getattr(usage, "requests", 0) or 0)


def _tool_args(part: Any) -> dict[str, Any]:
    try:
        args = part.args_as_dict()
    except Exception:
        return {}
    return args if isinstance(args, dict) else {}


def _tool_summary(event: FunctionToolResultEvent) -> str:
    part = event.part
    try:
        text = part.model_response_str() if hasattr(part, "model_response_str") else part.model_response()
    except Exception:
        text = str(getattr(part, "content", ""))
    text = " ".join(str(text).split())
    return text[:TOOL_SUMMARY_CHARS]


async def stream_advice(turn: ChatTurnRequest, deps: AdvisorDeps, model: Model, model_id: str, *,
                        timeout_s: float = TURN_TIMEOUT_S) -> AsyncIterator[Any]:
    """Run the advisor for one turn and yield ``delta``, ``tool_call`` and
    ``tool_result`` events as they happen, then ``final`` with the whole text
    and the usage, or ``error``. The caller adds ``done``. The turn stops
    after ``timeout_s``.

    ``final`` is what the deltas spelled out, except that a sentence claiming
    a run started, is running or is done (or an error is fixed) is replaced
    by an honest line unless the panel started a run in this turn
    (:func:`r3.agent.honesty.honest_reply`); the client shows ``final``."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout_s, 0.01)
    history = history_messages(turn.history)
    text_parts: list[str] = []
    new_part = False  # a text part started after a tool call: separate it from the text before
    result = None
    looked_up_runs = False  # the run history was read: a run it calls running is one it looked up
    try:
        async with advisor_agent.run_stream_events(
            turn.text, model=model, deps=deps, message_history=history,
            model_settings=ModelSettings(max_tokens=2048, timeout=60),
            usage_limits=UsageLimits(request_limit=8, tool_calls_limit=12),
        ) as events:
            iterator = aiter(events)
            while True:
                # The deadline guards each wait, never a yield: a timeout must
                # not fire while the consumer holds control.
                try:
                    async with asyncio.timeout_at(deadline):
                        event = await anext(iterator)
                except StopAsyncIteration:
                    break
                delta = ""
                if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                    delta, new_part = event.part.content, True
                elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    delta = event.delta.content_delta
                elif isinstance(event, FunctionToolCallEvent):
                    looked_up_runs = looked_up_runs or event.part.tool_name in ("recent_runs", "pipeline_stats")
                    yield ToolCallEvent(name=event.part.tool_name, args=_tool_args(event.part))
                elif isinstance(event, FunctionToolResultEvent):
                    yield ToolResultEvent(name=getattr(event.part, "tool_name", None) or "tool",
                                          summary=_tool_summary(event))
                elif isinstance(event, AgentRunResultEvent):
                    result = event.result
                if delta:
                    if new_part and text_parts and not "".join(text_parts).endswith("\n"):
                        delta = "\n\n" + delta
                    new_part = False
                    text_parts.append(delta)
                    yield DeltaEvent(text=delta)
    except Exception as exc:
        log.warning("Advisor turn failed on %s (%s)", model_id, _describe(exc))
        yield error_event(exc, model_id)
        return
    # The panel recorded the stats snapshot as a run in this turn.
    run_started = isinstance(deps.action, StatsAction) and deps.action.run is not None
    text = "".join(text_parts).strip()
    if not text and result is not None and isinstance(result.output, str) and result.output.strip():
        text = honest_reply(result.output.strip(), run_started=run_started, looked_up_runs=looked_up_runs)
        yield DeltaEvent(text=text)
    if not text:
        yield ErrorEvent(message="The AI provider returned an empty reply. Please try again.")
        return
    text = honest_reply(text, run_started=run_started, looked_up_runs=looked_up_runs)
    yield FinalEvent(text=text[:16000], model=model_id, usage=_usage(result))


def _describe(exc: BaseException) -> str:
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} {status}" if isinstance(status, int) else type(exc).__name__
