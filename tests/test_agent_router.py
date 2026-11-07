"""The intent router: the rules classifier and extractor, the fast path, the
LLM route (FunctionModel), the Jev route (FunctionModel answering with the
``provider_details`` layout TypeSafeModel produces, plus one contract test
against the real TypeSafeModel over a mocked transport), hybrid fallbacks."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider
from typesafe_sdk import AsyncTypeSafeClient

from r3 import db
from r3.agent.contracts import ChatMessage, RouteParams
from r3.agent.router import (
    PRICING,
    RESUME_HELP,
    IntentRouter,
    classify,
    fast_path,
    fix_command_typos,
    jev_field_confidence,
    jev_role_options,
    jev_route_type,
    llm_instructions,
    normalize_roles,
    normalize_tech,
    route_message,
    rules_params,
    rules_route,
)
from tests.fakes import test_settings as make_settings

MODEL_ID = "openrouter:openrouter/free"

FACETS = db.Facets(
    statuses=[db.FacetValue(value=v, count=10) for v in ("new", "applied", "skipped", "expired")],
    role_categories=[db.FacetValue(value=v, count=10) for v in ("AI/ML", "SRE", "DevOps", "Data", "AI", "ML", "Other")],
    location_kinds=[db.FacetValue(value=v, count=10) for v in ("remote-us", "remote-ca", "remote-ww", "onsite-us")],
    tech_tags=[db.FacetValue(value=v, count=10) for v in ("python", "kubernetes", "terraform", "go", "snowflake")],
)


def chat(*pairs: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in pairs]


LISTED = chat(("user", "show new sre jobs"),
              ("assistant", "The panel shows 3 jobs:\n#7 Senior SRE at Acme\n#9 SRE at Beta\n#11 Staff SRE at Gamma"))


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


# --- rules: intents ------------------------------------------------------------------------

@pytest.mark.parametrize("text, intent", [
    ("show stats", "stats"),
    ("statistics for the last 7 days", "stats"),
    ("how did the runs go?", "stats"),
    ("dashboard", "stats"),
    ("how many jobs do we have", "stats"),
    ("summary of the pipeline", "stats"),
    ("what's my progress", "stats"),
    ("show me new remote SRE jobs", "jobs"),
    ("list applied jobs", "jobs"),
    ("what jobs are available?", "jobs"),
    ("any unreviewed devops jobs?", "jobs"),
    ("open positions at Acme", "jobs"),
    ("find me remote sre jobs", "jobs"),
    ("how many remote sre jobs are there?", "jobs"),
    ("should I apply to #123?", "advice"),
    ("how do I apply to #12?", "advice"),
    ("what's the best way to prepare for an SRE interview?", "advice"),
    ("tell me about job 7", "jobs"),
    ("what do you think about job 7?", "advice"),
    ("Rewrite my summary line so it reads better for SRE roles.", "advice"),
    ("Where do things stand?", "stats"),
    ("hello there, I need help thinking about my search", "advice"),
])
def test_rules_intents(text, intent):
    decision = rules_route(text)
    assert decision.intent == intent, decision.reason
    assert decision.router == "rules" and decision.confidence is None and decision.reason


def test_classify_returns_a_reason():
    assert classify("show me sre jobs") == ("jobs", "asks to list or filter stored jobs")


# --- rules: parameters ---------------------------------------------------------------------

def test_roles_map_synonyms_to_canonical_buckets_without_facets():
    assert rules_params("find new machine learning jobs").roles == ["AI", "ML"]
    assert rules_params("site reliability roles").roles == ["SRE"]
    assert rules_params("platform engineer jobs").roles == ["DevOps"]
    assert rules_params("data engineer jobs").roles == ["Data"]
    assert rules_params("ai/ml or sre jobs").roles == ["AI", "ML", "SRE"]


def test_roles_map_onto_the_databases_categories():
    assert rules_params("show me ml jobs", facets=FACETS).roles == ["AI/ML", "AI", "ML"]
    assert rules_params("sre or devops jobs", facets=FACETS).roles == ["SRE", "DevOps"]


def test_params_extraction():
    p = rules_params("show 5 remote contract devops jobs in Canada paying 200k+ with kubernetes from the last 10 days",
                     facets=FACETS)
    assert p.limit == 5 and p.remote is True and p.location == "Canada" and p.employment == "contract"
    assert p.roles == ["DevOps"] and p.min_salary == 200_000 and p.tech == ["kubernetes"] and p.days == 10


@pytest.mark.parametrize("text, days", [
    ("jobs from this week", 7), ("jobs scraped today", 1), ("last 3 weeks", 21), ("past 48 hours", 2),
    ("posted in the last 30 days", 30), ("this month", 30), ("jobs 5 days old", 5), ("last 7d", 7),
])
def test_days(text, days):
    assert rules_params(text).days == days


@pytest.mark.parametrize("text, salary", [
    ("jobs paying 200k+", 200_000), ("over $180,000", 180_000), ("$150k remote jobs", 150_000),
    ("salary at least 175000", 175_000), ("jobs with a 401k match", None),
])
def test_salary(text, salary):
    assert rules_params(text).min_salary == salary


@pytest.mark.parametrize("text, ids", [
    ("show job 7", [7]),
    ("jobs 4521, 4522 and 4523", [4521, 4522, 4523]),
    ("tell me about #12 and #15", [12, 15]),
    ("show 3 jobs", []),
    ("jobs paying 150,000", []),
])
def test_job_ids(text, ids):
    assert rules_params(text).job_ids == ids


def test_limit_and_status():
    assert rules_params("top 3 new ML jobs").limit == 3
    assert rules_params("a couple of sre jobs").limit == 2
    assert rules_params("show applied jobs").status == "applied"
    assert rules_params("show skipped jobs").status == "skipped"
    assert rules_params("show all jobs").status == "all"
    assert rules_params("show applied jobs", intent="advice").status == ""


def test_an_explicit_search_beats_a_tech_tag():
    p = rules_params("any postings mentioning Kafka?")
    assert p.search == "kafka" and p.tech == []
    assert rules_params("kafka jobs").tech == ["kafka"]


def test_search_and_company():
    assert rules_params("open positions at Acme AI").search == "Acme AI"
    assert rules_params('jobs mentioning "event sourcing"').search == "event sourcing"


def test_remote_hybrid_onsite():
    assert rules_params("remote jobs").remote is True
    hybrid = rules_params("hybrid jobs")
    assert hybrid.remote is False and hybrid.location == "hybrid"
    assert rules_params("onsite jobs").location == "onsite"
    assert rules_params("remote or hybrid jobs").remote is None
    assert rules_params("jobs in the US").location == "US"
    assert rules_params("show us the jobs").location == ""  # "us" the pronoun


def test_a_company_name_is_not_a_role():
    p = rules_params("open positions at Acme AI")
    assert p.search == "Acme AI" and p.roles == []
    assert rules_params("jobs at Stripe for sre").roles == ["SRE"]
    assert rules_params("look at these jobs").search == ""


def test_numbers_that_are_not_ids():
    assert rules_params("no 3 is fine").job_ids == []
    assert rules_params("show me no. 31").job_ids == [31]
    assert fast_path("/Jobs Remote SRE").params.roles == ["SRE"]


# --- typos, other languages, one job, what a run did -------------------------------------------

def test_typos_are_corrected_only_for_commands_at_a_clause_start():
    assert fix_command_typos("stast please") == "stats please"
    assert fix_command_typos("thanks! now stast please") == "thanks! now stats please"
    assert fix_command_typos("show me the stast results") == "show me the stast results"  # not a clause start
    for words in ("status of the pipeline", "state of things", "stars and stripes", "sats"):
        assert fix_command_typos(words) == words
    assert rules_route("stast please").intent == "stats"
    assert rules_route("status of the pipeline").intent == "stats"


@pytest.mark.parametrize("text, intent, params", [
    ("muéstrame las estadísticas de esta semana", "stats", {"days": 7}),
    ("покажи статистику за неделю", "stats", {"days": 7}),
    ("покажи новые вакансии sre", "jobs", {"roles": ["SRE"]}),
    ("busca información sobre la empresa", "advice", {}),
])
def test_spanish_and_russian(text, intent, params):
    decision = rules_route(text)
    assert decision.intent == intent, decision.reason
    for key, value in params.items():
        assert getattr(decision.params, key) == value, key


def test_a_question_about_one_job_shows_it():
    decision = rules_route("is #7 still open?")
    assert decision.intent == "jobs" and decision.params.job_ids == [7]
    assert rules_route("#7 looks good").intent == "jobs"
    assert rules_route("no, skip #99").intent == "advice"  # setting a job aside is not asking to see it
    assert rules_route("should I apply to #7?").intent == "advice"


def test_contracts_are_jobs_but_a_contract_question_is_advice():
    decision = rules_route("any c2c devops contracts?")
    assert decision.intent == "jobs" and decision.params.roles == ["DevOps"] and decision.params.employment == "contract"
    assert rules_route("Is a 6-month C2C contract worth taking over a full-time offer?").intent == "advice"
    assert rules_route("how do contracts work?").intent == "advice"


@pytest.mark.parametrize("text", ["what did the last run count?", "show me the previous runs",
                                  "what did the runs find?"])
def test_what_a_run_did_is_stats(text):
    assert rules_route(text).intent == "stats"


def test_tech_uses_the_facet_spelling_and_skips_common_words():
    assert normalize_tech("k8s", FACETS) == "kubernetes"
    assert normalize_tech("golang", FACETS) == "go"
    assert rules_params("go ahead with the python jobs", facets=FACETS).tech == ["python"]


def test_normalize_roles():
    assert normalize_roles(["ml", "security"], None) == (["AI", "ML", "security"], [])
    assert normalize_roles(["Machine Learning", "sre", "Security"], FACETS) == (["AI/ML", "AI", "ML", "SRE"], ["Security"])
    assert normalize_roles(["devops"], db.Facets(role_categories=[db.FacetValue(value="SRE", count=1)])) == ([], ["devops"])


# --- fast path -----------------------------------------------------------------------------

def test_yes_is_no_command():
    """A bare "yes" confirms nothing: no proposal waits for one."""
    for history in ([], LISTED):
        decision = fast_path("yes", history)
        assert decision is None or (decision.intent == "advice" and not decision.params.job_ids)


@pytest.mark.parametrize("text, intent, reason", [
    ("/stats", "stats", "slash command /stats"),
    ("/jobs devops", "jobs", "slash command /jobs"),
    ("/ask how do I negotiate", "advice", "slash command /ask"),
    ("thanks!", "advice", "greeting or thanks"),
    ("hi", "advice", "greeting or thanks"),
])
def test_fast_path_commands_and_greetings(text, intent, reason):
    decision = fast_path(text)
    assert decision.intent == intent and decision.reason == reason


def test_slash_params():
    assert fast_path("/jobs remote sre").params.roles == ["SRE"]
    assert fast_path("/jobs 12 15").params.job_ids == []  # bare numbers are no ids
    assert fast_path("/scout remote sre") is None  # gone with scouting


# --- the LLM route -----------------------------------------------------------------------------

def output_call(args: dict[str, Any], seen: list | None = None) -> FunctionModel:
    """A model that answers every request by calling the output tool with ``args``."""

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append((messages, info))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(respond)


def failing(exc: Exception, calls: list | None = None) -> FunctionModel:
    def respond(messages, info: AgentInfo) -> ModelResponse:
        if calls is not None:
            calls.append(info)
        raise exc

    return FunctionModel(respond)


async def test_llm_route_maps_the_route_call():
    seen: list = []
    model = output_call({"intent": "jobs", "roles": ["site reliability"], "remote": True, "days": 900, "limit": 80,
                         "status": "Unreviewed", "employment": "Full-time", "tech": ["k8s"], "reason": "lists jobs"},
                        seen)
    router = IntentRouter(make_settings(router_mode="llm"), models={"llm": model})
    decision = await router.route("show remote unreviewed sre jobs", [], model_id=MODEL_ID, facets=FACETS)
    assert decision.router == "llm" and decision.intent == "jobs" and decision.reason == "lists jobs"
    p = decision.params
    assert p.roles == ["SRE"] and p.remote is True and p.days == 365 and p.limit == 50
    assert p.status == "new" and p.employment == "fulltime" and p.tech == ["kubernetes"]
    _, info = seen[0]
    assert info.model_settings.get("temperature") == 0.0


async def test_llm_route_sees_history_facets_and_date():
    seen: list = []
    router = IntentRouter(make_settings(router_mode="llm"), models={"llm": output_call({"intent": "jobs"}, seen)})
    await router.route("tell me about the first two", LISTED, model_id=MODEL_ID, facets=FACETS)
    messages, info = seen[0]
    prompt = messages[-1].parts[-1].content
    assert "#7 Senior SRE at Acme" in prompt and prompt.rstrip().endswith("tell me about the first two")
    for needle in ("AI/ML", "snowflake", "Intents:", '"should I apply to #12?" -> advice', "Today is"):
        assert needle in info.instructions


async def test_llm_route_grounds_ids():
    model = output_call({"intent": "jobs", "job_ids": [7, 999]})
    router = IntentRouter(make_settings(router_mode="llm"), models={"llm": model})
    decision = await router.route("tell me about the first one", LISTED, model_id=MODEL_ID)
    assert decision.params.job_ids == [7]  # 999 appears nowhere in the conversation


async def test_llm_route_follows_the_chat_model_without_a_router_model():
    seen: list = []
    router = IntentRouter(make_settings(router_mode="llm"), models={"chat": output_call({"intent": "stats"}, seen)})
    assert (await router.route("stats please", [], model_id=MODEL_ID)).intent == "stats"
    assert seen


async def test_llm_failure_falls_back_to_rules_without_the_provider_body():
    model = failing(ModelHTTPError(500, "openrouter/free", body="secret upstream body"))
    router = IntentRouter(make_settings(router_mode="llm"), models={"llm": model})
    decision = await router.route("show me new remote SRE jobs", [], model_id=MODEL_ID)
    assert decision.router == "fallback" and decision.intent == "jobs" and decision.params.roles == ["SRE"]
    assert "ModelHTTPError 500" in decision.reason and "secret" not in decision.reason


async def test_llm_without_a_key_falls_back():
    settings = make_settings(router_mode="llm", api_keys={"openai": "oa-test"})
    decision = await IntentRouter(settings).route("show stats", [], model_id=MODEL_ID)
    assert decision.router == "fallback" and decision.intent == "stats" and "ValueError" in decision.reason


# --- the Jev route -----------------------------------------------------------------------------

def jev(intent: str = "jobs", confidence: float | None = 0.9, *, roles=(), remote=None, status=None,
        employment=None, seen: list | None = None) -> FunctionModel:
    """A stand-in for TypeSafeModel: fills the output tool in one response and
    reports ``provider_details`` in TypeSafe's layout (confidence per field,
    the distribution of each pick-one)."""

    def respond(messages, info: AgentInfo) -> ModelResponse:
        if seen is not None:
            seen.append((messages, info))
        args = {"intent": intent, "roles": list(roles), "remote": remote, "status": status,
                "employment": employment}
        details: dict[str, Any] = {
            "confidence": {"roles": 0.8, "remote": 0.7, "status": 0.6, "employment": 0.9},
            "probabilities": {"intent": {intent: confidence or 0.0}, "remote": {"none": 0.7}},
            "scores": {},
        }
        if confidence is not None:
            details["confidence"]["intent"] = confidence
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)], provider_details=details)

    return FunctionModel(respond)


async def test_jev_route_fills_closed_sets_and_rules_fill_numbers():
    calls: list = []
    router = IntentRouter(make_settings(router_mode="jev"),
                          models={"jev": jev("jobs", 0.93, roles=["SRE"], remote="remote", status="new"),
                                  "llm": failing(RuntimeError("the LLM must not be called"), calls)})
    decision = await router.route("show me #12 and #15 from the last 7 days", [], model_id=MODEL_ID,
                                  facets=FACETS)
    assert decision.router == "jev" and decision.intent == "jobs" and decision.confidence == 0.93
    p = decision.params
    assert p.roles == ["SRE"] and p.remote is True and p.status == "new"  # from Jev
    assert p.job_ids == [12, 15] and p.days == 7  # from the rules
    assert not calls


async def test_jev_judges_the_latest_message_with_history():
    seen: list = []
    router = IntentRouter(make_settings(router_mode="jev"), models={"jev": jev("jobs", seen=seen)})
    await router.route("and the devops ones?", LISTED, model_id=MODEL_ID, facets=FACETS)
    messages, info = seen[0]
    assert isinstance(messages[0], ModelRequest) and messages[0].parts[0].content == "show new sre jobs"
    assert isinstance(messages[1], ModelResponse) and messages[1].parts[0].content.startswith("The panel shows")
    assert isinstance(messages[-1].parts[-1], UserPromptPart) and messages[-1].parts[-1].content == "and the devops ones?"
    schema = info.output_tools[0].parameters_json_schema
    assert [o["const"] for o in schema["properties"]["roles"]["items"]["anyOf"]] == ["AI/ML", "SRE", "DevOps", "Data",
                                                                                      "AI", "ML"]


def test_jev_route_type_and_role_options():
    assert jev_role_options(None) == ("SRE", "DevOps", "Data", "AI/ML")
    assert "Other" not in jev_role_options(FACETS)
    fields = jev_route_type().model_fields
    assert list(fields) == ["intent", "roles", "remote", "status", "employment"]
    assert jev_field_confidence({"confidence": {"intent": 0.7}}, "intent") == 0.7
    assert jev_field_confidence({"confidence": {}}, "intent") is None and jev_field_confidence(None, "intent") is None


async def test_jev_failure_falls_back_to_rules():
    router = IntentRouter(make_settings(router_mode="jev"), models={"jev": failing(UserError("unsupported field"))})
    decision = await router.route("show me new remote SRE jobs", [], model_id=MODEL_ID)
    assert decision.router == "fallback" and decision.intent == "jobs" and "Jev router failed (UserError)" in decision.reason


async def test_hybrid_takes_a_confident_jev_without_the_llm():
    calls: list = []
    router = IntentRouter(make_settings(router_mode="hybrid", jev_confidence=0.6),
                          models={"jev": jev("stats", 0.75), "llm": failing(RuntimeError("unused"), calls)})
    decision = await router.route("how are we doing", [], model_id=MODEL_ID)
    assert decision.router == "jev" and decision.intent == "stats" and not calls


async def test_hybrid_below_the_threshold_asks_the_llm():
    router = IntentRouter(make_settings(router_mode="hybrid", jev_confidence=0.6),
                          models={"jev": jev("stats", 0.3), "llm": output_call({"intent": "jobs", "reason": "lists"})})
    decision = await router.route("new sre jobs please", [], model_id=MODEL_ID)
    assert decision.router == "llm" and decision.intent == "jobs" and decision.confidence is None
    assert "Jev leaned stats at 0.30" in decision.reason


async def test_hybrid_without_a_jev_confidence_asks_the_llm():
    router = IntentRouter(make_settings(router_mode="hybrid"),
                          models={"jev": jev("jobs", None), "llm": output_call({"intent": "jobs"})})
    assert (await router.route("new sre jobs please", [], model_id=MODEL_ID)).router == "llm"


async def test_hybrid_jev_error_asks_the_llm_and_both_failing_fall_back():
    router = IntentRouter(make_settings(router_mode="hybrid"),
                          models={"jev": failing(UserError("x")), "llm": output_call({"intent": "stats"})})
    decision = await router.route("numbers please", [], model_id=MODEL_ID)
    assert decision.router == "llm" and "Jev failed (UserError)" in decision.reason
    router = IntentRouter(make_settings(router_mode="hybrid"),
                          models={"jev": failing(UserError("x")), "llm": failing(ModelHTTPError(429, "m"))})
    decision = await router.route("show stats", [], model_id=MODEL_ID)
    assert decision.router == "fallback" and decision.intent == "stats"
    assert "Jev failed (UserError)" in decision.reason and "ModelHTTPError 429" in decision.reason


# --- the rules' hint, the fast path in every mode ----------------------------------------------

async def test_the_llm_gets_the_rules_verdict_as_a_hint():
    seen: list = []
    router = IntentRouter(make_settings(router_mode="llm"), models={"llm": output_call({"intent": "jobs"}, seen)})
    await router.route("Show me new AI/ML jobs from this week", [], model_id=MODEL_ID)
    assert "A keyword pre-classifier read this message as: jobs because" in seen[0][1].instructions


@pytest.mark.parametrize("mode", ["llm", "jev", "hybrid"])
@pytest.mark.parametrize("text, intent", [("/stats", "stats"), ("/jobs remote sre", "jobs"), ("thanks", "advice")])
async def test_fast_path_skips_the_models_in_every_mode(mode, text, intent):
    calls: list = []
    model = failing(RuntimeError("no model call on the fast path"), calls)
    router = IntentRouter(make_settings(router_mode=mode), models={"llm": model, "jev": model, "chat": model})
    decision = await router.route(text, [], model_id=MODEL_ID)
    assert decision.router == "rules" and decision.intent == intent and not calls


async def test_route_message_convenience():
    decision = await route_message("tell me about job 123", [], settings=make_settings(router_mode="rules"),
                                   model_id=MODEL_ID)
    assert decision.intent == "jobs" and decision.params.job_ids == [123]


def test_route_params_normalise_status():
    assert RouteParams(status="PDF ready").status == ""
    assert RouteParams(status="Unreviewed").status == "new"
    assert RouteParams(status="bogus").status == ""


# --- contract with the real TypeSafeModel (offline: mocked HTTP transport) -----------------------

async def test_jev_route_type_against_the_real_typesafe_model():
    """Pins how TypeSafeModel reads JevRoute and lays out provider_details:
    the intent is one pick-one, roles fan out to one yes/no per option,
    remote/status/employment are pick-one-or-none."""
    sent: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        answers: dict[str, Any] = {}
        for name, question in body["questions"].items():
            if question["type"] == "choice":
                options = list(question["criteria"])
                pick = {"intent": "jobs", "remote": "remote", "status": "none", "employment": "none"}[name]
                answers[name] = {"type": "choice", "choice": pick, "confidence": 0.8 if name == "intent" else 0.6,
                                 "probabilities": {o: (0.9 if o == pick else 0.1 / (len(options) - 1)) for o in options}}
            else:
                answers[name] = {"type": "noul", "noul": {"roles.SRE": 0.93}.get(name, 0.04)}
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": answers,
                                          "usage": {"input_tokens": 120, "output_tokens": 6}})

    client = AsyncTypeSafeClient(api_key="ts-test", transport=httpx2.MockTransport(handler))
    model = TypeSafeModel("jev-latest", provider=TypeSafeProvider(typesafe_client=client))
    router = IntentRouter(make_settings(router_mode="jev"), models={"jev": model})
    with override_allow_model_requests(True):
        decision = await router.route("show me #12 and #15, the remote sre ones", LISTED, model_id=MODEL_ID)
    assert len(sent) == 1  # one Jev request per turn
    questions = sent[0]["questions"]
    assert set(questions) == {"intent", "roles.SRE", "roles.DevOps", "roles.Data", "roles.AI/ML", "remote", "status",
                              "employment"}
    assert questions["intent"]["criteria"]["jobs"].startswith("Show, list, filter or count the jobs")
    assert set(questions["intent"]["criteria"]) == {"advice", "jobs", "stats"}
    assert "none" in questions["remote"]["criteria"]
    state = sent[0]["state"]
    assert state["text"].startswith("show me #12") and state["history"][0] == {"user": "show new sre jobs"}
    assert decision.router == "jev" and decision.intent == "jobs" and decision.confidence == 0.8
    p = decision.params
    assert p.roles == ["SRE"] and p.remote is True and p.status == "" and p.employment == ""
    assert p.job_ids == [12, 15]


def test_response_shape_helpers_do_not_read_text_parts():
    # A text part in a Jev response would be a contract break; the confidence reader ignores everything else.
    response = ModelResponse(parts=[TextPart("x")], provider_details={"confidence": {"intent": 1.5}})
    assert jev_field_confidence(response.provider_details, "intent") is None


# --- QA: a resume request, a price question (L1/L5, L13) ----------------------------------

PASTED = chat(("user", "Here is my background: 6 years of SRE at Bluefjord, Kubernetes, Terraform, on-call lead."),
              ("assistant", "Thanks, that is a strong SRE background."))


@pytest.mark.parametrize("text, history", [
    ("make me a resume from this", PASTED),
    ("Help me build my resume", []),
    ("can you create a CV from my background?", []),
    ("write my resume", []),
    ("turn this into a resume", PASTED),
    ("update my resume", []),
    ("I need a resume, can you build me one from the above?", PASTED),
    ("make a pdf of my resume", []),  # their own resume: its page has Download PDF
])
def test_a_resume_request_is_resume_help(text, history):
    decision = rules_route(text, history)
    assert (decision.intent, decision.reason) == ("advice", RESUME_HELP), decision
    assert decision.params.job_ids == []


@pytest.mark.parametrize("text", ["how much does this cost?", "What does R3 cost?", "is it free?", "pricing?",
                                  "do I have to pay for this?"])
def test_a_price_question_is_advice(text):
    decision = rules_route(text)
    assert (decision.intent, decision.reason) == ("advice", PRICING), decision


@pytest.mark.parametrize("text, intent", [
    ("how much do remote SRE jobs pay?", "jobs"),
    ("show me cloud cost optimization jobs", "jobs"),
])
def test_pay_and_cost_words_about_jobs_are_not_price_questions(text, intent):
    decision = rules_route(text)
    assert decision.intent == intent and decision.reason != PRICING


@pytest.mark.parametrize("mode", ["llm", "jev", "hybrid"])
@pytest.mark.parametrize("text, answer, reason", [
    ("make me a resume from this", "jobs", RESUME_HELP),
    ("Help me build my resume", "stats", RESUME_HELP),
    ("how much does this cost?", "stats", PRICING),
])
async def test_models_cannot_turn_these_into_workflows(mode, text, answer, reason):
    model = output_call({"intent": answer, "job_ids": [12, 15]})
    router = IntentRouter(make_settings(router_mode=mode, jev_confidence=0.5),
                          models={"llm": model, "jev": jev(answer, 0.95)})
    decision = await router.route(text, PASTED, model_id=MODEL_ID)
    assert (decision.intent, decision.reason) == ("advice", reason), decision
    assert decision.params.job_ids == []


# --- OQ-7: jobs are named as links ------------------------------------------------------------

def test_the_llm_router_is_told_jobs_are_links():
    text = llm_instructions(FACETS)
    assert "#<id> Position at Company" not in text
