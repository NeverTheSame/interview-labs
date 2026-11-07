"""pydantic-evals evaluators for the Controller's two suites.

Routing (one case = one operator message, ``evals/cases.py``). The task
returns :class:`RouteOutcome`; the case's expectation is a
:class:`RouteExpectation`:

- :class:`IntentMatch`: the route's intent is the expected one. Also emits the
  predicted intent as the ``intent`` label, which the confusion matrix reads.
- :class:`ParamsSubset`: every parameter the case names matches the route's
  ``RouteParams`` (lists as sets: roles through ``normalise_role``, tech
  lower-cased, job ids as ints; ints and bools exact; ``search`` and
  ``location`` by case-insensitive containment; other strings case-insensitive).
- :class:`ActionKind`: the panel action has the expected kind.
- :class:`RouterUsed`: which path decided (rules | llm | jev | fallback) as the
  ``router`` label, and Jev's intent confidence as the ``confidence`` score.
  pydantic-evals records ``set_eval_attribute``/``increment_eval_metric`` only
  while the task runs (the task-run context is reset before evaluators run), so
  the harness records the same facts as attributes and metrics; this evaluator
  puts them in the report table, where the Averages row turns the label into
  the share of turns per path (the fallback rate).

Advisor (one case = one chat turn with a scripted route, ``evals/cases_advisor.py``):
:class:`EventOrder`, :class:`AdvisorAction`, :class:`MentionsRequired`,
:class:`DeltasSpellFinal`, :class:`PlainText` and :class:`AtMostSentences` are
deterministic and run offline and live; :class:`JevJudge` (live, with
``TYPESAFE_API_KEY``) and pydantic-evals' ``LLMJudge`` (live) grade the answer.

An evaluator returns ``{}`` for a case it has nothing to check, so every pass
rate counts only the cases that state that expectation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, TypedDict

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, EvaluatorOutput

from evals.cases import normalise_role
from r3.agent.contracts import RouteDecision, RouteParams

# --- routing ---------------------------------------------------------------------------


class RouteInputs(TypedDict):
    """What the operator typed, and the conversation before it."""

    text: str
    history: list[dict[str, str]]  # [{"role": "user" | "assistant", "content": ...}]


class RouteOutcome(TypedDict):
    """What the Controller made of it: the route, and the kind of panel action."""

    decision: RouteDecision
    action_kind: str | None


@dataclass(frozen=True)
class RouteExpectation:
    """A case's expected route (``Case.expected_output``); ``None`` means "not checked"."""

    intent: str
    params: Mapping[str, Any] = field(default_factory=dict)
    action: str | None = None


ROUTING_CHECKS: tuple[str, ...] = ("IntentMatch", "ParamsSubset", "ActionKind")
CONTAINMENT_PARAMS = frozenset({"search", "location"})


def _as_set(key: str, values: Any) -> set[Any]:
    items = list(values or [])
    if key == "roles":
        return {normalise_role(str(v)) for v in items}  # deduped: "AI" and "ML" are both "AI/ML"
    if key == "job_ids":
        return {int(v) for v in items}
    return {str(v).strip().lower() for v in items}


def param_matches(key: str, want: Any, got: Any) -> bool:
    """One expected parameter against the route's value (see the module docstring)."""
    if isinstance(want, (list, tuple, set, frozenset)):
        return isinstance(got, (list, tuple, set, frozenset)) and _as_set(key, got) == _as_set(key, want)
    if isinstance(want, bool) or want is None:
        return got is want
    if isinstance(want, int):
        return isinstance(got, int) and not isinstance(got, bool) and got == want
    if isinstance(want, str):
        have = got.strip().lower() if isinstance(got, str) else ""
        return want.strip().lower() in have if key in CONTAINMENT_PARAMS else have == want.strip().lower()
    return bool(got == want)


def param_mismatches(expected: Mapping[str, Any], params: RouteParams) -> list[str]:
    """The expected parameters the route got wrong, as "key: want X, got Y"."""
    out: list[str] = []
    for key, want in expected.items():
        if key not in RouteParams.model_fields:
            out.append(f"{key}: not a RouteParams field")
        elif not param_matches(key, want, got := getattr(params, key)):
            out.append(f"{key}: want {want!r}, got {got!r}")
    return out


def _expectation(ctx: EvaluatorContext[Any, Any, Any]) -> RouteExpectation | None:
    expected = ctx.expected_output
    return expected if isinstance(expected, RouteExpectation) else None


def _check(ok: bool, reason: str) -> EvaluationReason:
    return EvaluationReason(value=ok, reason=None if ok else reason)


@dataclass(repr=False)
class IntentMatch(Evaluator[RouteInputs, Any, Any]):
    """``decision.intent == expected.intent``; the predicted intent is the ``intent`` label."""

    def evaluate(self, ctx: EvaluatorContext[RouteInputs, Any, Any]) -> EvaluatorOutput:
        got = ctx.output["decision"].intent
        expected = _expectation(ctx)
        if expected is None:
            return {"intent": got}
        return {"IntentMatch": _check(got == expected.intent, f"expected {expected.intent}, got {got}"), "intent": got}


@dataclass(repr=False)
class ParamsSubset(Evaluator[RouteInputs, Any, Any]):
    """Every parameter the case names matches; the reason lists the keys that failed."""

    def evaluate(self, ctx: EvaluatorContext[RouteInputs, Any, Any]) -> EvaluatorOutput:
        expected = _expectation(ctx)
        if expected is None or not expected.params:
            return {}
        failed = param_mismatches(expected.params, ctx.output["decision"].params)
        return {"ParamsSubset": _check(not failed, "; ".join(failed))}


@dataclass(repr=False)
class ActionKind(Evaluator[RouteInputs, Any, Any]):
    """The panel action ``plan_action`` returned (against StubDB + FakeRuns) has the expected kind."""

    def evaluate(self, ctx: EvaluatorContext[RouteInputs, Any, Any]) -> EvaluatorOutput:
        expected = _expectation(ctx)
        if expected is None or expected.action is None:
            return {}
        got = ctx.output["action_kind"]
        return {"ActionKind": _check(got == expected.action, f"expected {expected.action}, got {got}")}


@dataclass(repr=False)
class RouterUsed(Evaluator[RouteInputs, Any, Any]):
    """The path that decided (``router`` label) and Jev's intent confidence (``confidence`` score)."""

    def evaluate(self, ctx: EvaluatorContext[RouteInputs, Any, Any]) -> EvaluatorOutput:
        decision: RouteDecision = ctx.output["decision"]
        out: dict[str, Any] = {"router": decision.router}
        if decision.confidence is not None:
            out["confidence"] = float(decision.confidence)
        return out


ROUTING_EVALUATORS: tuple[type[Evaluator[Any, Any, Any]], ...] = (
    IntentMatch, ParamsSubset, ActionKind, RouterUsed,
)


# --- advisor ---------------------------------------------------------------------------


class AdvisorInputs(TypedDict):
    """One chat turn and the route the case scripts for it."""

    messages: list[dict[str, str]]  # the conversation, ending with the user's message
    decision: dict[str, Any]  # the scripted RouteDecision, as JSON
    runs: str  # "ok", or "down" for a run service that refuses every run


class AdvisorOutcome(TypedDict):
    """What the turn streamed, reduced to what the checks and the judges read."""

    events: list[str]  # event types in order
    action_kind: str | None
    panel: str  # what the panel on the right shows, in words
    requires: list[str]  # what the answer must mention, derived from the action
    answer: str  # the final text ("" when the turn ended in an error)
    deltas: str  # the delta events joined
    error: str
    usage: dict[str, int]


@dataclass(frozen=True)
class AdvisorExpectation:
    action: str  # the panel action kind the scripted route leads to


_INTENT_WORD = {"jobs": "job", "stats": "stat", "advice": ""}


def required_mentions(action: Mapping[str, Any] | None) -> list[str]:
    """What an answer must mention for the panel action it narrates: the
    first job listed (its link), a stats snapshot's total and new counts, and
    for an action that could not run, the workflow and any ``#id`` its reason
    names (an id the user typed). Jobs are named as links, never ``#id`` (OQ-7)."""
    if not action:
        return []
    kind = action.get("kind")
    if kind == "jobs":
        jobs = action.get("jobs") or []
        return [f"/jobs/{jobs[0]['id']}"] if jobs else []
    if kind == "stats":
        snapshot = action["snapshot"]
        return [str(snapshot["total_jobs"]), str(snapshot["new_jobs"])]
    if kind == "unavailable":
        word = _INTENT_WORD.get(str(action.get("intent")), "")
        return [w for w in (word, *re.findall(r"#\d+", str(action.get("reason", "")))) if w]
    return []


def mentions(text: str, item: str) -> bool:
    """Case-insensitive; a number must stand alone ("100" is not in "1000"),
    written plain or with thousands separators, and a job ``#7`` (or its link
    ``/jobs/7``) is not ``#70`` (``/jobs/70``)."""
    if item.isdigit():
        spellings = {item, f"{int(item):,}"}
        return any(re.search(rf"(?<![\d,.]){re.escape(s)}(?![\d,]|\.\d)", text) for s in spellings)
    if re.fullmatch(r"#\d+|/jobs/\d+", item):
        return re.search(rf"{re.escape(item)}(?!\d)", text) is not None
    return item.lower() in text.lower()


_EVENT_ORDER = re.compile(r"route action (?:(?:tool_call|tool_result|delta) )*final done")
_MARKDOWN = re.compile(r"^\s{0,3}#{1,6}\s|\*\*[^*\n]+\*\*|__[^_\n]+__|^\s*\|.*\|\s*$", re.M)


def sentence_count(text: str) -> int:
    """Sentences, roughly: each list item counts as one, prose splits after . ! ?"""
    count = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(?:[-*•]|\d+[.)])\s", line):
            count += 1
        else:
            count += len([part for part in re.split(r"(?<=[.!?])\s+", line) if part.strip()])
    return count


@dataclass(repr=False)
class EventOrder(Evaluator[AdvisorInputs, Any, Any]):
    """route → action → (tool_call | tool_result | delta)* → final → done."""

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        seen = " ".join(ctx.output["events"])
        return _check(_EVENT_ORDER.fullmatch(seen) is not None, f"events were: {seen}")


@dataclass(repr=False)
class AdvisorAction(Evaluator[AdvisorInputs, Any, Any]):
    """The scripted route leads to the expected panel action."""

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        expected = ctx.expected_output
        if not isinstance(expected, AdvisorExpectation):
            return {}
        got = ctx.output["action_kind"]
        return _check(got == expected.action, f"expected {expected.action}, got {got}")


@dataclass(repr=False)
class MentionsRequired(Evaluator[AdvisorInputs, Any, Any]):
    """The answer mentions what the action requires (see :func:`required_mentions`)."""

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        answer = ctx.output["answer"]
        missing = [item for item in ctx.output["requires"] if not mentions(answer, item)]
        return _check(bool(answer) and not missing, "no answer" if not answer else f"missing: {', '.join(missing)}")


@dataclass(repr=False)
class DeltasSpellFinal(Evaluator[AdvisorInputs, Any, Any]):
    """The streamed deltas add up to the final text."""

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        answer, deltas = ctx.output["answer"], ctx.output["deltas"].strip()
        return _check(bool(answer) and deltas == answer, "the deltas differ from the final text")


@dataclass(repr=False)
class PlainText(Evaluator[AdvisorInputs, Any, Any]):
    """No Markdown headings, bold or tables (lists with "-" are fine)."""

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        hit = _MARKDOWN.search(ctx.output["answer"])
        return _check(hit is None, f"markdown: {hit.group(0).strip()!r}" if hit else "")


@dataclass(repr=False)
class AtMostSentences(Evaluator[AdvisorInputs, Any, Any]):
    """At most ``limit`` sentences (list items count as one each)."""

    limit: int = 8

    def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        n = sentence_count(ctx.output["answer"])
        return _check(n <= self.limit, f"{n} sentences")


ADVISOR_CHECKS: tuple[type[Evaluator[Any, Any, Any]], ...] = (
    EventOrder, AdvisorAction, MentionsRequired, DeltasSpellFinal, PlainText, AtMostSentences,
)


class Verdict(BaseModel):
    """One turn in R3, the controller of an operator's job search: the operator's message, what the panel
    on the right shows, and the assistant's reply on the left. Only the panel starts runs and lists jobs; the
    assistant narrates what the panel shows and answers questions."""

    claims_action_it_did_not_take: bool = Field(description=(
        "Does the assistant's reply claim that the assistant itself started a run, applied, or changed "
        "something, rather than describing what the panel shows?"))
    names_the_run_or_jobs: bool = Field(description=(
        "Does the assistant's reply name the run id or the jobs (as links to /jobs/<id>) that the panel shows?"))
    answers_the_question: bool = Field(description="Does the assistant's reply answer the operator's message?")


_JEV_JUDGE: Agent[None, Verdict] = Agent(output_type=Verdict, defer_model_check=True, name="r3-evals-jev-judge",
                                          retries=0)


def jev_state(inputs: AdvisorInputs, outcome: AdvisorOutcome) -> str:
    """The text Jev judges: the whole turn in one piece."""
    return (f"Operator's message: {inputs['messages'][-1]['content']}\n"
            f"What the panel on the right shows: {outcome['panel']}\n"
            f"Assistant's reply: {outcome['answer']}")


@dataclass(repr=False)
class JevJudge(Evaluator[AdvisorInputs, Any, Any]):
    """Jev answers three yes/no questions about the reply (:class:`Verdict`) in
    one request; each assertion's reason carries Jev's confidence, read from
    ``provider_details["confidence"]``. "Names the run or jobs" is only asked
    of turns whose panel lists jobs."""

    model: Model | None = None  # a TypeSafeModel, built by the harness

    async def evaluate(self, ctx: EvaluatorContext[AdvisorInputs, Any, Any]) -> EvaluatorOutput:
        outcome = ctx.output
        if not outcome["answer"]:
            return {"JevNoFalseClaims": EvaluationReason(False, "no answer to judge")}
        result = await _JEV_JUDGE.run(jev_state(ctx.inputs, outcome), model=self.model)
        verdict = result.output
        confidence = (result.response.provider_details or {}).get("confidence") or {}

        def said(name: str, ok: bool) -> EvaluationReason:
            value = confidence.get(name)
            return EvaluationReason(ok, f"confidence {value:.2f}" if isinstance(value, (int, float)) else None)

        out: dict[str, Any] = {
            "JevNoFalseClaims": said("claims_action_it_did_not_take", not verdict.claims_action_it_did_not_take),
            "JevAnswers": said("answers_the_question", verdict.answers_the_question),
        }
        if outcome["action_kind"] == "jobs":
            out["JevNamesRunOrJobs"] = said("names_the_run_or_jobs", verdict.names_the_run_or_jobs)
        return out

    def build_serialization_arguments(self) -> dict[str, Any]:
        arguments = super().build_serialization_arguments()
        if isinstance(arguments.get("model"), Model):
            arguments["model"] = arguments["model"].model_id
        return arguments
