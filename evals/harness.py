"""The eval harness: datasets, tasks, and the numbers reports and thresholds
are built from.

Routing. :func:`build_routing_dataset` turns ``evals/cases.py`` into a
pydantic-evals ``Dataset``. :func:`routing_task` returns the task for one
router mode: it routes the message with ``route_message`` (the facets come
from the test database, as ``Advisor.stream_turn`` fetches them), then plans
the panel action with ``plan_action`` against a fresh ``StubDB`` + ``FakeRuns``
(the test database: one new AI/ML job, #7 Senior ML Engineer at Acme AI), and
returns ``{"decision", "action_kind"}``. Offline, the model-backed modes run on
:class:`ScriptedRouterModels`, FunctionModels that answer every case with its
expected route: they prove the plumbing (prompt, parsing, grounding, the
hybrid fallback, the panel action), not the models, so the meaningful offline
baseline is ``rules``. Live, the models are built from the environment
(``ControllerSettings.from_env()``).

``IntentRouter`` does not report token usage, so the task wraps every model it
injects in :class:`RecordingModel`, a pass-through ``WrapperModel`` that sums
``ModelResponse.usage`` per case and keeps Jev's intent confidence (including
the confidences hybrid mode rejects). Latency is the wall time of the
``route_message`` call.

Advisor. :func:`build_advisor_dataset` turns ``evals/cases_advisor.py`` into a
dataset; :func:`advisor_task` runs ``Advisor.stream_turn`` end to end with the
case's route scripted (:class:`ScriptedRouter`), against the same test
database and run service. Offline the answer comes from :func:`narrator_model`;
live from the chat model, graded by ``LLMJudge`` and, with
``TYPESAFE_API_KEY``, :class:`~evals.evaluators.JevJudge`.

Thresholds. :data:`THRESHOLDS` are the committed floors, measured first and
set to the measured rate minus 0.03, rounded down to 0.05
(:func:`honest_floor`). They apply to full-dataset runs, on the cases not
tagged "hard"; hard cases are reported separately."""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Callable, Literal, Mapping, Sequence

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models import Model, ModelRequestParameters, override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_evals import Case, Dataset, increment_eval_metric, set_eval_attribute
from pydantic_evals.evaluators import (
    ConfusionMatrixEvaluator,
    Evaluator,
    LLMJudge,
    MaxDuration,
    PrecisionRecallEvaluator,
)
from pydantic_evals.reporting import EvaluationReport
from pydantic_evals.reporting.analyses import ConfusionMatrix, ScalarResult

from evals.cases import CASES, RoutingCase, normalise_role
from evals.cases_advisor import ADVISOR_CASES, ADVISOR_RUBRIC, AdvisorCase
from evals.evaluators import (
    ADVISOR_CHECKS,
    ROUTING_CHECKS,
    ROUTING_EVALUATORS,
    AdvisorExpectation,
    AdvisorInputs,
    AdvisorOutcome,
    JevJudge,
    RouteExpectation,
    RouteInputs,
    RouteOutcome,
    required_mentions,
)
from r3.agent.actions import fetch_facets, plan_action
from r3.agent.advisor import TURN_TIMEOUT_S
from r3.agent.contracts import INTENTS, ChatMessage, ChatTurnRequest, RouteDecision, RouteParams
from r3.agent.models import build_model
from r3.agent.router import jev_field_confidence, route_message
from r3.agent.service import Advisor, effective_default_model
from r3.settings import PROVIDER_KEYS, ControllerSettings
from tests.fakes import FakeRuns
from tests.fakes import test_settings as _offline_settings
from tests.test_routes import StubDB

Mode = Literal["rules", "llm", "jev", "hybrid"]
MODES: tuple[Mode, ...] = ("rules", "llm", "jev", "hybrid")
HARD = "hard"

# --- thresholds ---------------------------------------------------------------------------

THRESHOLD_METRICS: tuple[str, ...] = ("intent_accuracy", "params_pass_rate", "action_pass_rate")
CHECK_FOR_METRIC: dict[str, str] = dict(zip(THRESHOLD_METRICS, ROUTING_CHECKS, strict=True))


def honest_floor(measured: float, margin: float = 0.03, step: float = 0.05) -> float:
    """The committed floor for a measured rate: minus ``margin``, rounded down to ``step``."""
    return round(math.floor((measured - margin) / step + 1e-9) * step, 2)


# The floors were measured offline on the full dataset before being committed:
# rules scored 1.000 on intent, params and action; llm/jev/hybrid run on the
# scripted models offline and measured 1.000 everywhere (plumbing).
_SCRIPTED = {"intent_accuracy": 0.95, "params_pass_rate": 0.95, "action_pass_rate": 0.95}
THRESHOLDS: dict[str, dict[str, float]] = {
    "rules": {"intent_accuracy": 0.95, "params_pass_rate": 0.95, "action_pass_rate": 0.95},
    "llm": dict(_SCRIPTED),
    "jev": dict(_SCRIPTED),
    "hybrid": dict(_SCRIPTED),
}
# Live runs of the model modes have not been measured yet (no provider keys where
# the suite was built). Until they are, a model router must at least match the
# rules' first measured floor (the rules have since been tuned on this dataset, so
# their current floor would be no fair bar for an untuned model); replace these
# with honest_floor(measured) after the first live run.
LIVE_THRESHOLDS: dict[str, dict[str, float]] = {
    mode: {"intent_accuracy": 0.85, "params_pass_rate": 0.95, "action_pass_rate": 0.85}
    for mode in ("llm", "jev", "hybrid")
}

# What the rules router gets wrong today, measured: the intent it returns, then
# why. The report marks these "known"; any other failure is new.
RULES_KNOWN_FAILURES: dict[str, str] = {}

JEV_USD_PER_MILLION_INPUT = 0.042  # TypeSafe's published price; output tokens are free


# --- routing: inputs, dataset ---------------------------------------------------------------


def route_inputs(case: RoutingCase) -> RouteInputs:
    return {"text": case.text, "history": [{"role": t.role, "content": t.content} for t in case.history]}


def _inputs_key(inputs: RouteInputs) -> str:
    return json.dumps([inputs["text"], inputs["history"]], sort_keys=True)


def expectation(case: RoutingCase) -> RouteExpectation:
    return RouteExpectation(intent=case.intent, params=dict(case.params), action=case.action)


def select_cases(cases: Sequence[RoutingCase], tags: Sequence[str] = (), limit: int | None = None) -> list[RoutingCase]:
    """The cases carrying any of ``tags`` (all when none), the first ``limit`` of them."""
    wanted = {t.strip() for t in tags if t.strip()}
    chosen = [c for c in cases if not wanted or wanted & set(c.tags)]
    return chosen[:limit] if limit else chosen


def build_routing_dataset(cases: Sequence[RoutingCase] = CASES, *, name: str = "r3-routing",
                          confidence_curve: bool = False) -> Dataset[RouteInputs, Any, dict[str, Any]]:
    """One pydantic-evals case per routing case: inputs ``{"text", "history"}``,
    the expectation as ``expected_output``, ``{"intent", "tags", "hard"}`` as
    metadata. Report evaluators: the intent confusion matrix, and with
    ``confidence_curve`` the precision-recall of Jev's confidence against a
    correct intent."""
    report_evaluators: list[Any] = [ConfusionMatrixEvaluator(
        predicted_from="labels", predicted_key="intent", expected_from="metadata", expected_key="intent",
        title="Intent: expected (rows) x predicted (columns)")]
    if confidence_curve:
        report_evaluators.append(PrecisionRecallEvaluator(
            score_key="confidence", positive_from="assertions", positive_key="IntentMatch",
            title="Jev confidence vs a correct intent"))
    return Dataset(
        name=name,
        cases=[Case(name=c.name, inputs=route_inputs(c), expected_output=expectation(c),
                    metadata={"intent": c.intent, "tags": list(c.tags), "hard": HARD in c.tags}) for c in cases],
        evaluators=[evaluator() for evaluator in ROUTING_EVALUATORS],
        report_evaluators=report_evaluators,
    )


# --- routing: models ----------------------------------------------------------------------------


class RecordingModel(WrapperModel):
    """A pass-through model that records what each response cost and, for Jev,
    the intent confidence it reported."""

    def __init__(self, wrapped: Model) -> None:
        super().__init__(wrapped)
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.confidences: list[float] = []

    async def request(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                      model_request_parameters: ModelRequestParameters) -> ModelResponse:
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        self.requests += 1
        self.input_tokens += response.usage.input_tokens or 0
        self.output_tokens += response.usage.output_tokens or 0
        if (confidence := jev_field_confidence(response.provider_details, "intent")) is not None:
            self.confidences.append(confidence)
        return response

    def usage(self) -> dict[str, int]:
        return {"requests": self.requests, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}


def default_jev_confidence(case: RoutingCase) -> float:
    """What the scripted Jev reports: sure, except of hard and ambiguous cases,
    which it reports below hybrid's default floor (0.6) so hybrid asks the LLM."""
    return 0.45 if {"hard", "ambiguous"} & set(case.tags) else 0.9


def _consts(schema: Any) -> list[Any]:
    """Every ``const`` in a JSON schema: the options of a Literal field."""
    if isinstance(schema, dict):
        found = [schema["const"]] if "const" in schema else []
        return found + [c for value in schema.values() for c in _consts(value)]
    if isinstance(schema, list):
        return [c for item in schema for c in _consts(item)]
    return []


class ScriptedRouterModels:
    """Offline stand-ins for the router's models, built per case: an LLM that
    answers with the case's expected route as a ``RouteCall``, and a Jev that
    fills the ``JevRoute`` closed sets from it (roles, remote, status,
    employment) with TypeSafe's
    ``provider_details`` layout, its intent confidence from ``jev_confidence``.
    Numbers and free text in jev mode still come from the rules, as they do live."""

    def __init__(self, cases: Sequence[RoutingCase], *,
                 jev_confidence: Callable[[RoutingCase], float] = default_jev_confidence) -> None:
        self._cases = {_inputs_key(route_inputs(case)): case for case in cases}
        self.jev_confidence = jev_confidence

    def __call__(self, inputs: RouteInputs) -> dict[str, Model]:
        case = self._cases.get(_inputs_key(inputs))
        if case is None:
            raise KeyError(f"No scripted route for {inputs['text']!r}")
        return {"llm": FunctionModel(self._llm(case), model_name="scripted-llm"),
                "jev": FunctionModel(self._jev(case), model_name="scripted-jev")}

    @staticmethod
    def _llm(case: RoutingCase) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            args = {"intent": case.intent, **case.params, "reason": f"scripted route for {case.name}"}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

        return respond

    def _jev(self, case: RoutingCase) -> Callable[[list[ModelMessage], AgentInfo], ModelResponse]:
        confidence = self.jev_confidence(case)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            fields = info.output_tools[0].parameters_json_schema["properties"]
            wanted = {normalise_role(r) for r in case.params.get("roles", [])}
            status, employment = case.params.get("status"), case.params.get("employment")
            remote = case.params.get("remote")
            args = {
                "intent": case.intent,
                "roles": [o for o in _consts(fields["roles"]) if normalise_role(o) in wanted],
                "remote": "remote" if remote is True else "onsite_or_hybrid" if remote is False else None,
                "status": status if status in _consts(fields["status"]) else None,
                "employment": employment if employment in _consts(fields["employment"]) else None,
            }
            details = {"confidence": {"intent": confidence, "roles": 0.9, "remote": 0.9, "status": 0.9,
                                      "employment": 0.9},
                       "probabilities": {"intent": {case.intent: confidence}}, "scores": {}}
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)], provider_details=details)

        return respond


ModelSource = Mapping[str, Model] | Callable[[RouteInputs], Mapping[str, Model]] | None


def router_llm_id(settings: ControllerSettings) -> str:
    """The model the router's LLM path uses: ``R3_ROUTER_MODEL``, else the turn's chat model."""
    return settings.router_model or effective_default_model(settings)


def jev_model_id(settings: ControllerSettings) -> str:
    model_id = settings.typesafe_model
    return model_id if ":" in model_id else f"typesafe:{model_id}"


def missing_live_keys(mode: str, settings: ControllerSettings) -> list[str]:
    """The provider keys a live run of ``mode`` needs and does not have."""
    missing: list[str] = []
    if mode in ("llm", "hybrid"):
        provider = router_llm_id(settings).split(":", 1)[0]
        if not settings.provider_ready(provider):
            missing.append(f"{PROVIDER_KEYS.get(provider, provider + ' key')} (for {router_llm_id(settings)})")
    if mode in ("jev", "hybrid") and not settings.jev_ready:
        missing.append("TYPESAFE_API_KEY")
    return missing


def live_settings(mode: str, env: Mapping[str, str] | None = None) -> ControllerSettings:
    return replace(ControllerSettings.from_env(env), router_mode=mode)  # type: ignore[arg-type]


def live_models(mode: str, settings: ControllerSettings) -> dict[str, Model]:
    """The real models a live run injects (so their usage is recorded)."""
    models: dict[str, Model] = {}
    if mode in ("llm", "hybrid"):
        models["llm"] = build_model(router_llm_id(settings), settings)
    if mode in ("jev", "hybrid"):
        models["jev"] = build_model(jev_model_id(settings), settings)
    return models


# --- routing: the task --------------------------------------------------------------------------


def routing_task(mode: Mode, settings: ControllerSettings, models: ModelSource = None, *, facets: bool = True,
                 record_tokens: bool = False) -> Callable[[RouteInputs], Any]:
    """The task for one router mode: ``route_message`` then ``plan_action``
    against a fresh StubDB + FakeRuns. ``models`` are injected into the router
    (``{"llm": ..., "jev": ...}``), or built per case by a callable (the
    scripted models). Records the router, confidence, reason, latency and
    usage as attributes, and latency, model requests, fallbacks (and, live,
    tokens) as metrics."""
    settings = replace(settings, router_mode=mode)
    model_id = effective_default_model(settings)

    async def route(inputs: RouteInputs) -> RouteOutcome:
        database, runs = StubDB(), FakeRuns()
        found = await fetch_facets(database) if facets else None
        history = [ChatMessage.model_validate(m) for m in inputs["history"]]
        chosen = models(inputs) if callable(models) else (models or {})
        recorders = {name: RecordingModel(model) for name, model in chosen.items()}
        started = time.perf_counter()
        decision = await route_message(inputs["text"], history, settings=settings, model_id=model_id, facets=found,
                                       models=recorders)
        route_ms = (time.perf_counter() - started) * 1000
        action = await plan_action(decision, database=database, runs=runs, settings=settings, model_id=model_id,
                                   text=inputs["text"], facets=found)
        _record_route(decision, route_ms, recorders, record_tokens)
        return {"decision": decision, "action_kind": action.kind}

    route.__name__ = route.__qualname__ = f"route_{mode}"
    return route


def _record_route(decision: RouteDecision, route_ms: float, recorders: Mapping[str, RecordingModel],
                  record_tokens: bool) -> None:
    jev = recorders.get("jev")
    set_eval_attribute("router", decision.router)
    set_eval_attribute("confidence", decision.confidence)
    set_eval_attribute("reason", decision.reason)
    set_eval_attribute("route_ms", round(route_ms, 3))
    set_eval_attribute("usage", {name: rec.usage() for name, rec in recorders.items() if rec.requests})
    set_eval_attribute("jev_confidence", jev.confidences[-1] if jev is not None and jev.confidences else None)
    increment_eval_metric("route_ms", route_ms)
    increment_eval_metric("model_requests", sum(rec.requests for rec in recorders.values()))
    if decision.router == "fallback":
        increment_eval_metric("fallback", 1)
    if record_tokens:
        increment_eval_metric("input_tokens", sum(rec.input_tokens for rec in recorders.values()))
        increment_eval_metric("output_tokens", sum(rec.output_tokens for rec in recorders.values()))


# --- routing: the summary -----------------------------------------------------------------------


@dataclass
class Rate:
    passed: int = 0
    total: int = 0

    def add(self, ok: bool) -> None:
        self.total += 1
        self.passed += bool(ok)

    @property
    def value(self) -> float | None:
        return self.passed / self.total if self.total else None

    def __str__(self) -> str:
        return f"{self.value:.3f} ({self.passed}/{self.total})" if self.total else "n/a"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


@dataclass
class CaseResult:
    name: str
    text: str
    history: list[dict[str, str]]
    tags: tuple[str, ...]
    expected: RouteExpectation
    decision: RouteDecision | None  # None when the task raised
    action_kind: str | None
    checks: dict[str, bool]  # the ROUTING_CHECKS this case states, passed or not
    reasons: dict[str, str]
    route_ms: float | None = None
    usage: dict[str, dict[str, int]] = field(default_factory=dict)
    jev_confidence: float | None = None
    error: str = ""

    @property
    def hard(self) -> bool:
        return HARD in self.tags

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def predicted(self) -> str:
        return self.decision.intent if self.decision else "error"

    @property
    def router(self) -> str:
        return self.decision.router if self.decision else "error"


def _stated_checks(expected: RouteExpectation) -> list[str]:
    stated = (True, bool(expected.params), expected.action is not None)
    return [check for check, says in zip(ROUTING_CHECKS, stated, strict=True) if says]


@dataclass
class RoutingSummary:
    """Everything the report and the thresholds read from one mode's run."""

    mode: str
    live: bool
    results: list[CaseResult]
    full_dataset: bool
    confusion: ConfusionMatrix | None = None
    scalars: list[ScalarResult] = field(default_factory=list)

    def rates(self, *, hard: bool = False) -> dict[str, Rate]:
        rates = {metric: Rate() for metric in THRESHOLD_METRICS}
        for result in self.results:
            if result.hard != hard:
                continue
            for metric, check in CHECK_FOR_METRIC.items():
                if check in result.checks:
                    rates[metric].add(result.checks[check])
        return rates

    def per_intent(self) -> dict[str, tuple[Rate, Rate]]:
        """Intent accuracy per expected intent: (scored cases, hard cases)."""
        table = {intent: (Rate(), Rate()) for intent in INTENTS}
        for result in self.results:
            table[result.expected.intent][1 if result.hard else 0].add(result.predicted == result.expected.intent)
        return table

    def routers(self) -> Counter[str]:
        return Counter(result.router for result in self.results)

    def model_turns(self) -> list[CaseResult]:
        """Turns a model was asked about (everything but the fast path, in the model modes)."""
        return [r for r in self.results if self.mode != "rules" and r.router not in ("rules", "error")]

    def fallback_rates(self) -> dict[str, Rate]:
        """Of the model turns: decided by the rules after a model failure, and
        (hybrid) decided by the LLM because Jev was unsure or failed."""
        rules, llm = Rate(), Rate()
        for result in self.model_turns():
            rules.add(result.router == "fallback")
            llm.add(result.router == "llm")
        return {"to_rules": rules, "to_llm": llm} if self.mode == "hybrid" else {"to_rules": rules}

    def jev_confidences(self) -> list[float]:
        return [r.jev_confidence for r in self.results if r.jev_confidence is not None]

    def latencies(self, *, model_turns_only: bool = False) -> list[float]:
        chosen = self.model_turns() if model_turns_only else self.results
        return [r.route_ms for r in chosen if r.route_ms is not None]

    def usage(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = {}
        for result in self.results:
            for name, used in result.usage.items():
                into = totals.setdefault(name, {"requests": 0, "input_tokens": 0, "output_tokens": 0})
                for key, value in used.items():
                    into[key] = into.get(key, 0) + value
        return totals

    def jev_cost_usd(self) -> float:
        return self.usage().get("jev", {}).get("input_tokens", 0) * JEV_USD_PER_MILLION_INPUT / 1_000_000

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]

    def known(self, result: CaseResult) -> bool:
        return self.mode == "rules" and result.name in RULES_KNOWN_FAILURES

    def thresholds(self) -> dict[str, float]:
        return (LIVE_THRESHOLDS if self.live and self.mode != "rules" else THRESHOLDS).get(self.mode, {})

    def violations(self) -> list[str]:
        """Scored rates below this mode's floors; none for a subset run (the
        floors are for the full dataset)."""
        if not self.full_dataset:
            return []
        rates = self.rates()
        out: list[str] = []
        for metric, floor in self.thresholds().items():
            value = rates[metric].value
            if value is not None and value + 1e-9 < floor:
                out.append(f"{self.mode}: {metric} {value:.3f} < {floor:.2f}")
        return out


def summarize_routing(report: EvaluationReport[Any, Any, Any], cases: Sequence[RoutingCase], *, mode: str,
                      live: bool, full_dataset: bool) -> RoutingSummary:
    by_name = {case.name: case for case in cases}
    results: list[CaseResult] = []
    for rc in report.cases:
        case = by_name[rc.name]
        checks = {name: bool(result.value) for name, result in rc.assertions.items() if name in ROUTING_CHECKS}
        reasons = {name: result.reason or "" for name, result in rc.assertions.items()
                   if name in ROUTING_CHECKS and result.reason}
        for failure in rc.evaluator_failures:  # an evaluator that raised fails its check
            if failure.name in ROUTING_CHECKS:
                checks[failure.name], reasons[failure.name] = False, failure.error_message
        results.append(CaseResult(
            name=case.name, text=case.text, history=route_inputs(case)["history"], tags=case.tags,
            expected=expectation(case), decision=rc.output["decision"], action_kind=rc.output["action_kind"],
            checks=checks, reasons=reasons, route_ms=rc.attributes.get("route_ms"),
            usage=rc.attributes.get("usage") or {}, jev_confidence=rc.attributes.get("jev_confidence")))
    for failure in report.failures:
        case = by_name[failure.name]
        expected = expectation(case)
        results.append(CaseResult(
            name=case.name, text=case.text, history=route_inputs(case)["history"], tags=case.tags,
            expected=expected, decision=None, action_kind=None,
            checks={check: False for check in _stated_checks(expected)},
            reasons={"IntentMatch": failure.error_message}, error=failure.error_message))
    order = {case.name: i for i, case in enumerate(cases)}
    results.sort(key=lambda r: order[r.name])
    confusion = next((a for a in report.analyses if isinstance(a, ConfusionMatrix)), None)
    scalars = [a for a in report.analyses if isinstance(a, ScalarResult)]
    return RoutingSummary(mode=mode, live=live, results=results, full_dataset=full_dataset, confusion=confusion,
                          scalars=scalars)


@dataclass
class RoutingRun:
    summary: RoutingSummary
    report: EvaluationReport[Any, Any, Any]
    settings: ControllerSettings
    models: dict[str, str]  # role -> model id, for the report header


async def run_routing(mode: Mode, *, live: bool = False, cases: Sequence[RoutingCase] = CASES,
                      settings: ControllerSettings | None = None, models: ModelSource = None,
                      concurrency: int | None = None, progress: bool = False) -> RoutingRun:
    """Evaluate one router mode. Offline (default): scripted models for
    llm/jev/hybrid, no model request can leave the process. Live: settings and
    models from the environment (the caller checks :func:`missing_live_keys`)."""
    settings = settings or (live_settings(mode) if live else _offline_settings(router_mode=mode))
    if models is None and mode != "rules":
        models = live_models(mode, settings) if live else ScriptedRouterModels(cases)
    described: dict[str, str] = {}
    if mode in ("llm", "hybrid"):
        described["llm"] = router_llm_id(settings) if live else "scripted FunctionModel (expected route)"
    if mode in ("jev", "hybrid"):
        described["jev"] = jev_model_id(settings) if live else "scripted FunctionModel (expected route)"
    dataset = build_routing_dataset(cases, name=f"r3-routing-{mode}", confidence_curve=mode in ("jev", "hybrid"))
    task = routing_task(mode, settings, models, record_tokens=live)
    with override_allow_model_requests(live):
        report = await dataset.evaluate(task, name=f"routing {mode} ({'live' if live else 'offline'})",
                                        max_concurrency=concurrency, progress=progress)
    summary = summarize_routing(report, cases, mode=mode, live=live,
                                full_dataset=len(cases) == len(CASES) and list(cases) == list(CASES))
    return RoutingRun(summary=summary, report=report, settings=settings, models=described)


# --- advisor --------------------------------------------------------------------------------


class ScriptedRouter:
    """Stands in for ``IntentRouter``: every turn gets the case's route."""

    def __init__(self, decision: RouteDecision) -> None:
        self.decision = decision

    async def route(self, text: str, history: Sequence[ChatMessage], *, model_id: str,
                    facets: Any = None) -> RouteDecision:
        return self.decision


_NARRATION_HEAD = "The router read this message as"


def _facts(narration: str) -> list[str]:
    """A short plain answer from the facts the advisor's narration carries."""
    sentences: list[str] = []
    if m := re.search(r"asked for an? (\w+) workflow but it could not run: (.+?)\.(?:\s|$)", narration):
        sentences.append(f"The {m.group(1)} workflow could not run: {m.group(2)}.")
        if hint := re.search(r"Hint: (.+?)(?:\s+Explain\b|$)", narration, re.S):
            sentences.append(hint.group(1).strip())
        return sentences
    run_ids = list(dict.fromkeys(re.findall(r"\b(r3-[a-z]+-[\w-]+)", narration)))
    numbers = dict(re.findall(r'"(total_jobs|new_jobs|applied_jobs)":\s*(\d+)', narration))
    if numbers:
        sentences.append(f"The snapshot shows {numbers.get('total_jobs', '?')} jobs in total: "
                         f"{numbers.get('new_jobs', '?')} new and {numbers.get('applied_jobs', '?')} applied.")
        if run_ids:
            sentences.append(f"It is recorded as run {run_ids[0]}.")
    jobs = re.findall(r"^(\[[^\]\n]+\]\(/jobs/\d+\))\s*$", narration, re.M)  # the narration's job links
    if jobs:
        sentences.append(f"The panel lists {'; '.join(jobs[:3])}.")
    return sentences or ["Lead with the systems you have kept running and the results you shipped; ask me for the "
                         "numbers whenever you want them."]


def narrator_model() -> FunctionModel:
    """The offline advisor: calls ``job_details`` when the narration asks for
    it, or ``portal_overview`` for plain advice, then streams a plain answer
    built only from what the narration says (so an answer that lacks the run
    id means the narration lacked it)."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[Any]:
        narration = (info.instructions or "").rsplit(_NARRATION_HEAD, 1)[-1]
        last = messages[-1]
        returned = isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts)
        if not returned:
            if "job_details" in narration and (job := re.search(r"^\[[^\]\n]+\]\(/jobs/(\d+)\)", narration, re.M)):
                yield {0: DeltaToolCall(name="job_details", json_args=json.dumps({"job_id": int(job.group(1))}),
                                        tool_call_id="evals-call-1")}
                return
            if "No workflow was started" in narration:
                yield {0: DeltaToolCall(name="portal_overview", json_args="{}", tool_call_id="evals-call-1")}
                return
        words = " ".join(_facts(narration)).split(" ")
        third = max(1, len(words) // 3)
        for i in range(0, len(words), third):
            yield " ".join(words[i:i + third]) + (" " if i + third < len(words) else "")

    return FunctionModel(stream_function=stream, model_name="scripted-narrator")


def advisor_inputs(case: AdvisorCase) -> AdvisorInputs:
    decision = RouteDecision(intent=case.route["intent"], params=RouteParams(**case.route.get("params", {})),
                             router="rules", reason="route scripted by the evals")
    messages = [{"role": t.role, "content": t.content} for t in case.history] + [{"role": "user", "content": case.text}]
    return {"messages": messages, "decision": decision.model_dump(mode="json"), "runs": case.runs}


def describe_panel(action: Mapping[str, Any] | None) -> str:
    """What the panel shows, in words, for the judges."""
    if not action:
        return "nothing (the turn failed before an action)"
    kind = action["kind"]
    if kind == "jobs":
        listed = "; ".join(f"{j['position']} at {j['company']} (/jobs/{j['id']})" for j in action["jobs"][:5])
        return f"a job list: {action['title']}" + (f" ({listed})" if listed else "")
    if kind == "stats":
        s = action["snapshot"]
        run = f", recorded as run {action['run']['id']}" if action.get("run") else ""
        return f"a statistics snapshot: {s['total_jobs']} jobs, {s['new_jobs']} new, {s['applied_jobs']} applied{run}"
    if kind == "unavailable":
        return f"the {action['intent']} workflow could not run: {action['reason']}"
    return "nothing: no workflow for this message, the assistant answers"


def advisor_task(settings: ControllerSettings, models: Mapping[str, Model] | None = None, *,
                 record_tokens: bool = False) -> Callable[[AdvisorInputs], Any]:
    """Run ``Advisor.stream_turn`` for one turn with the case's route scripted,
    against a fresh StubDB and a FakeRuns that answers (or refuses, "down")."""

    async def advise(inputs: AdvisorInputs) -> AdvisorOutcome:
        decision = RouteDecision.model_validate(inputs["decision"])
        router: Any = ScriptedRouter(decision)
        advisor = Advisor(settings, models=dict(models or {}), router=router)
        runs = FakeRuns(configured=inputs["runs"] != "down", reason="the run service is down (scripted by the evals)")
        turn = ChatTurnRequest(messages=inputs["messages"])  # type: ignore[arg-type]
        events = [event async for event in advisor.stream_turn(turn, StubDB(), runs)]
        action_event = next((e for e in events if e.type == "action"), None)
        action = action_event.action.model_dump(mode="json") if action_event is not None else None
        final = next((e for e in events if e.type == "final"), None)
        error = next((e for e in events if e.type == "error"), None)
        usage = final.usage.model_dump() if final is not None else {}
        increment_eval_metric("advisor_requests", usage.get("requests", 0))
        if record_tokens:
            increment_eval_metric("input_tokens", usage.get("input_tokens", 0))
            increment_eval_metric("output_tokens", usage.get("output_tokens", 0))
        return {
            "events": [e.type for e in events],
            "action_kind": action["kind"] if action else None,
            "panel": describe_panel(action),
            "requires": required_mentions(action),
            "answer": final.text if final is not None else "",
            "deltas": "".join(e.text for e in events if e.type == "delta"),
            "error": error.message if error is not None else "",
            "usage": usage,
        }

    return advise


def build_advisor_dataset(cases: Sequence[AdvisorCase] = ADVISOR_CASES, *,
                          judges: Sequence[Evaluator[Any, Any, Any]] = ()) -> Dataset[AdvisorInputs, Any, Any]:
    """The advisor cases with the deterministic checks, the advisor's own turn
    deadline as ``MaxDuration``, plus ``judges`` (live)."""
    return Dataset(
        name="r3-advisor",
        cases=[Case(name=c.name, inputs=advisor_inputs(c), expected_output=AdvisorExpectation(action=c.action),
                    metadata={"tags": list(c.tags)}) for c in cases],
        evaluators=[*(check() for check in ADVISOR_CHECKS), MaxDuration(seconds=TURN_TIMEOUT_S), *judges],
    )


def judge_model_id(settings: ControllerSettings, env: Mapping[str, str] | None = None) -> str:
    """``R3_EVALS_JUDGE_MODEL``, else the chat model."""
    raw = (env if env is not None else os.environ).get("R3_EVALS_JUDGE_MODEL", "").strip()
    return raw or effective_default_model(settings)


def live_judges(settings: ControllerSettings) -> list[Evaluator[Any, Any, Any]]:
    """LLMJudge on ``R3_EVALS_JUDGE_MODEL`` (default: the chat model), and Jev when its key is set."""
    judges: list[Evaluator[Any, Any, Any]] = [LLMJudge(
        rubric=ADVISOR_RUBRIC, model=build_model(judge_model_id(settings), settings), include_input=True,
        assertion={"evaluation_name": "LLMJudge", "include_reason": True})]
    if settings.jev_ready:
        judges.append(JevJudge(model=build_model(jev_model_id(settings), settings)))
    return judges


@dataclass
class AdvisorRun:
    report: EvaluationReport[Any, Any, Any]
    live: bool
    model: str
    judges: list[str]

    def rates(self) -> dict[str, Rate]:
        rates: dict[str, Rate] = {}
        for rc in self.report.cases:
            for name, result in rc.assertions.items():
                rates.setdefault(name, Rate()).add(bool(result.value))
            for failure in rc.evaluator_failures:
                rates.setdefault(failure.name, Rate()).add(False)
        for _ in self.report.failures:
            rates.setdefault("task", Rate()).add(False)
        return rates

    def failures(self) -> list[tuple[str, str, str]]:
        """(case, check, reason) for every failed check."""
        out = [(f.name, "task", f.error_message) for f in self.report.failures]
        for rc in self.report.cases:
            out += [(rc.name, name, r.reason or "") for name, r in rc.assertions.items() if not r.value]
            out += [(rc.name, f.name, f.error_message) for f in rc.evaluator_failures]
        return out

    def violations(self) -> list[str]:
        """Offline, every deterministic check must pass; live runs are graded, not gated."""
        if self.live:
            return []
        return [f"advisor: {case} {check} failed ({reason})" for case, check, reason in self.failures()]


def missing_advisor_keys(settings: ControllerSettings) -> list[str]:
    return [] if settings.available_models() else ["OPENROUTER_API_KEY or OPENAI_API_KEY (for the chat model)"]


async def run_advisor(*, live: bool = False, cases: Sequence[AdvisorCase] = ADVISOR_CASES,
                      concurrency: int | None = None, progress: bool = False) -> AdvisorRun:
    """The advisor suite. The route is scripted and the router runs in rules
    mode, so only the advisor is under test."""
    if live:
        settings = replace(ControllerSettings.from_env(), router_mode="rules")
        models: dict[str, Model] = {}
        judges = live_judges(settings)
        model = effective_default_model(settings)
    else:
        settings = _offline_settings(router_mode="rules")
        models, judges, model = {"chat": narrator_model()}, [], "scripted narrator (FunctionModel)"
    dataset = build_advisor_dataset(cases, judges=judges)
    with override_allow_model_requests(live):
        report = await dataset.evaluate(advisor_task(settings, models, record_tokens=live),
                                        name=f"advisor ({'live' if live else 'offline'})",
                                        max_concurrency=concurrency, progress=progress)
    return AdvisorRun(report=report, live=live, model=model, judges=[j.get_default_evaluation_name() for j in judges])


def usage_tokens(summary_usage: Mapping[str, Mapping[str, int]]) -> tuple[int, int]:
    """(input, output) tokens over every model role."""
    return (sum(u.get("input_tokens", 0) for u in summary_usage.values()),
            sum(u.get("output_tokens", 0) for u in summary_usage.values()))


__all__ = [
    "CHECK_FOR_METRIC", "HARD", "JEV_USD_PER_MILLION_INPUT", "LIVE_THRESHOLDS", "MODES", "RULES_KNOWN_FAILURES",
    "THRESHOLDS", "THRESHOLD_METRICS", "AdvisorRun", "CaseResult", "Mode", "Rate", "RecordingModel", "RoutingRun",
    "RoutingSummary", "ScriptedRouter", "ScriptedRouterModels", "advisor_inputs", "advisor_task",
    "build_advisor_dataset", "build_routing_dataset", "describe_panel", "honest_floor", "jev_model_id",
    "live_judges", "live_models", "live_settings", "missing_advisor_keys", "missing_live_keys", "narrator_model",
    "percentile", "route_inputs", "router_llm_id", "routing_task", "run_advisor", "run_routing", "select_cases",
    "summarize_routing",
]
