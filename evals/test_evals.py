"""The evals as tests, offline (part of ``uv run pytest -q``): the rules router
meets its committed floors and fails only where the baseline says it does;
every stated panel action holds for the rules on the cases not tagged hard;
the scripted llm/jev/hybrid models route perfectly through the real router
(the plumbing), hybrid hands a doubtful Jev to the LLM; the advisor suite
passes its structural checks; the judges and the CLI work."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from evals.cases import CASES
from evals.cases_advisor import ADVISOR_CASES, ADVISOR_RUBRIC
from evals.evaluators import (
    ADVISOR_CHECKS,
    JevJudge,
    mentions,
    param_matches,
    required_mentions,
    sentence_count,
)
from evals.harness import (
    HARD,
    LIVE_THRESHOLDS,
    MODES,
    RULES_KNOWN_FAILURES,
    THRESHOLD_METRICS,
    THRESHOLDS,
    ScriptedRouterModels,
    advisor_task,
    build_advisor_dataset,
    honest_floor,
    narrator_model,
    route_inputs,
    routing_task,
    run_advisor,
    run_routing,
    select_cases,
)
from evals.run import main
from r3.agent.contracts import INTENTS, RouteParams
from tests.fakes import test_settings as make_settings

ACTION_KINDS = {"advice", "jobs", "stats", "unavailable"}
PLUMBING = ("jobs-new-ai-week", "jobs-detail", "stats-runs-week", "advice-positioning")


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


def by_name(*names: str) -> list[Any]:
    return [case for case in CASES if case.name in names]


# --- the dataset --------------------------------------------------------------------------


def test_the_dataset_is_well_formed():
    names = [case.name for case in CASES]
    assert len(names) == len(set(names)) and len(CASES) >= 40
    for case in CASES:
        assert case.intent in INTENTS, case.name
        assert set(case.params) <= set(RouteParams.model_fields), case.name
        RouteParams(**case.params)  # every expected value is a valid parameter
        assert case.action is None or case.action in ACTION_KINDS, case.name
        roles = [turn.role for turn in case.history]
        assert roles == ["user", "assistant"] * (len(roles) // 2), case.name
    assert set(RULES_KNOWN_FAILURES) <= set(names)


def test_thresholds_cover_every_mode_and_follow_the_rule():
    assert set(THRESHOLDS) == set(MODES) and set(LIVE_THRESHOLDS) == {"llm", "jev", "hybrid"}
    for floors in [*THRESHOLDS.values(), *LIVE_THRESHOLDS.values()]:
        assert set(floors) == set(THRESHOLD_METRICS)
    assert [honest_floor(v) for v in (0.912, 1.0, 0.95, 0.905)] == [0.85, 0.95, 0.9, 0.85]


# --- the rules router ---------------------------------------------------------------------


async def test_rules_mode_meets_its_thresholds_with_no_new_failures():
    summary = (await run_routing("rules")).summary
    assert summary.full_dataset and not summary.violations(), summary.violations()
    new = {r.name: r.reasons for r in summary.failures() if r.name not in RULES_KNOWN_FAILURES}
    assert not new, f"new routing failures (fix them, or record them in RULES_KNOWN_FAILURES): {new}"
    assert {r.router for r in summary.results} == {"rules"}


def _action_case(case: Any) -> Any:
    if case.name in RULES_KNOWN_FAILURES:
        return pytest.param(case, id=case.name, marks=pytest.mark.xfail(reason=RULES_KNOWN_FAILURES[case.name]))
    return pytest.param(case, id=case.name)


@pytest.mark.parametrize("case", [_action_case(c) for c in CASES if c.action and HARD not in c.tags])
async def test_rules_panel_action(case):
    outcome = await routing_task("rules", make_settings())(route_inputs(case))
    assert outcome["action_kind"] == case.action, outcome["decision"].reason


# --- the model modes' plumbing, on scripted models --------------------------------------------


@pytest.mark.parametrize("mode", ["llm", "jev", "hybrid"])
async def test_scripted_models_route_perfectly_through_the_router(mode):
    cases = by_name(*PLUMBING)
    summary = (await run_routing(mode, cases=cases)).summary
    assert [r.name for r in summary.failures()] == [], [(r.name, r.reasons) for r in summary.failures()]
    assert all(rate.value == 1.0 for rate in summary.rates().values() if rate.total)
    assert {r.router for r in summary.results} == {"jev" if mode in ("jev", "hybrid") else "llm"}
    assert summary.usage()["jev" if mode != "llm" else "llm"]["requests"] == len(cases)
    assert not summary.full_dataset and not summary.violations()


async def test_hybrid_asks_the_llm_when_jev_is_unsure():
    cases = by_name(*PLUMBING)
    doubtful = ScriptedRouterModels(cases, jev_confidence=lambda case: 0.3)
    summary = (await run_routing("hybrid", cases=cases, models=doubtful)).summary
    assert {r.router for r in summary.results} == {"llm"} and not summary.failures()
    assert summary.fallback_rates()["to_llm"].value == 1.0 and summary.fallback_rates()["to_rules"].value == 0.0
    assert summary.jev_confidences() == [0.3] * len(cases)  # recorded although hybrid set Jev aside
    assert all("Jev leaned" in r.decision.reason for r in summary.results)


async def test_a_failing_model_is_counted_as_a_fallback():
    def broken(messages: Any, info: AgentInfo) -> ModelResponse:
        raise RuntimeError("provider down")

    cases = by_name("jobs-new-ai-week", "stats-numbers")
    summary = (await run_routing("llm", cases=cases, models={"llm": FunctionModel(broken)})).summary
    assert {r.router for r in summary.results} == {"fallback"} and not summary.failures()
    assert summary.fallback_rates()["to_rules"].value == 1.0


# --- the advisor --------------------------------------------------------------------------------


async def test_the_advisor_passes_its_structural_checks_offline():
    run = await run_advisor()
    assert not run.failures() and not run.violations(), run.failures()
    rates = run.rates()
    assert set(rates) == {check.__name__ for check in ADVISOR_CHECKS} | {"MaxDuration"}
    assert all(rate.total == len(ADVISOR_CASES) for rate in rates.values())
    kinds = {rc.name: rc.output["action_kind"] for rc in run.report.cases}
    assert set(kinds.values()) == {"jobs", "stats", "advice"}


async def test_the_advisor_checks_catch_an_answer_that_skips_the_jobs():
    async def terse(messages: Any, info: AgentInfo):
        yield "Done."

    case = next(c for c in ADVISOR_CASES if c.name == "jobs-list")
    report = await build_advisor_dataset([case]).evaluate(
        advisor_task(make_settings(router_mode="rules"), {"chat": FunctionModel(stream_function=terse)}),
        progress=False)
    assertions = report.cases[0].assertions
    assert assertions["EventOrder"].value is True and assertions["MentionsRequired"].value is False
    assert "/jobs/7" in assertions["MentionsRequired"].reason


async def test_the_judges_read_the_turn():
    seen: list[str] = []

    def jev(messages: Any, info: AgentInfo) -> ModelResponse:
        args = {"claims_action_it_did_not_take": False, "names_the_run_or_jobs": True, "answers_the_question": True}
        details = {"confidence": {"claims_action_it_did_not_take": 0.8, "names_the_run_or_jobs": 0.9,
                                  "answers_the_question": 0.7}}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)], provider_details=details)

    def llm(messages: Any, info: AgentInfo) -> ModelResponse:
        seen.append(messages[-1].parts[-1].content)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,
                                                 {"reason": "names the jobs", "pass": True, "score": 1.0})])

    case = next(c for c in ADVISOR_CASES if c.name == "jobs-list")
    judges = [JevJudge(model=FunctionModel(jev)), LLMJudge(rubric=ADVISOR_RUBRIC, model=FunctionModel(llm),
                                                             include_input=True)]
    report = await build_advisor_dataset([case], judges=judges).evaluate(
        advisor_task(make_settings(router_mode="rules"), {"chat": narrator_model()}), progress=False)
    a = report.cases[0].assertions
    assert a["JevNoFalseClaims"].value and a["JevNoFalseClaims"].reason == "confidence 0.80"
    assert a["JevNamesRunOrJobs"].value and a["JevAnswers"].value and a["LLMJudge"].value
    assert "/jobs/7" in seen[0] and "at most 8 sentences" in seen[0]


# --- helpers ------------------------------------------------------------------------------------


def test_param_matching():
    assert param_matches("roles", ["AI/ML"], ["AI", "ML"]) and param_matches("roles", ["ai"], ["AI/ML"])
    assert not param_matches("roles", ["SRE"], ["SRE", "DevOps"])
    assert param_matches("tech", ["Kubernetes"], ["kubernetes"]) and param_matches("job_ids", [9, 7], [7, 9])
    assert param_matches("search", "kafka", "Kafka streams") and param_matches("location", "canada", "Canada")
    assert param_matches("status", "Applied", "applied") and not param_matches("status", "new", "new, applied")
    assert not param_matches("limit", 3, None) and not param_matches("remote", True, None)
    assert not param_matches("remote", True, 1)


def test_answer_helpers():
    jobs = {"kind": "jobs", "jobs": [{"id": 7, "position": "Senior ML Engineer"}]}
    stats = {"kind": "stats", "snapshot": {"total_jobs": 1200, "new_jobs": 40}}
    down = {"kind": "unavailable", "intent": "jobs", "reason": "The database did not answer"}
    assert required_mentions(jobs) == ["/jobs/7"]  # a link, never "#7" (OQ-7)
    assert required_mentions(stats) == ["1200", "40"] and required_mentions(down) == ["job"]
    assert mentions("1,200 jobs, 40 new.", "1200") and mentions("It has 40.", "40") and not mentions("1400", "40")
    assert mentions("#7 Senior ML Engineer", "#7") and not mentions("#70 Data Lead", "#7")
    assert mentions("[Senior ML Engineer](/jobs/7)", "/jobs/7") and not mentions("[Data Lead](/jobs/70)", "/jobs/7")
    assert sentence_count("One. Two!\n- a list item\n\nThree?") == 4


def test_select_cases():
    assert {c.name for c in select_cases(CASES, ["fast-path"])} >= {"jobs-slash", "stats-slash"}
    assert len(select_cases(CASES, limit=5)) == 5 and select_cases(CASES, ["no-such-tag"]) == []


# --- the command line -----------------------------------------------------------------------------


def test_the_cli_writes_a_report(tmp_path, capsys):
    path = tmp_path / "rules.md"
    assert main(["--mode", "rules", "--report", str(path)]) == 0
    text = path.read_text(encoding="utf-8")
    for needle in ("## Mode: rules (offline)", "| Intent accuracy | ", "### Confusion matrix", "### Router usage",
                   "### Latency", "### Failures", "### All cases"):
        assert needle in text, needle
    out = capsys.readouterr().out
    assert "Thresholds for rules" in out and "floor 0.95" in out and "All thresholds met." in out


def test_the_cli_on_a_subset_and_on_bad_input(tmp_path, capsys):
    path = tmp_path / "fast.md"
    assert main(["--mode", "hybrid", "--tags", "fast-path", "--report", str(path)]) == 0
    assert "not gated (subset)" in path.read_text(encoding="utf-8")
    assert main(["--mode", "rules", "--tags", "no-such-tag"]) == 2
    assert "no-such-tag" in capsys.readouterr().err


def test_live_refuses_to_run_without_keys(tmp_path, monkeypatch, capsys):
    for key in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "TYPESAFE_API_KEY", "R3_CHAT_MODEL", "R3_ROUTER_MODEL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env to find
    assert main(["--mode", "hybrid", "--live"]) == 2
    err = capsys.readouterr().err
    assert "Refusing to run live" in err and "TYPESAFE_API_KEY" in err and "OPENROUTER_API_KEY" in err
