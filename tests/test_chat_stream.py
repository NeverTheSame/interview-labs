"""The chat turn end to end: POST /api/chat/stream on the real app with the
stub database, the in-memory run service and the real Advisor running on
FunctionModel/TestModel. Covers the event order, the narration the advisor
is given, model resolution errors, safe provider errors, the model picker."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel

from r3.agent.advisor import error_event, stream_advice
from r3.agent.contracts import AdviceAction, ChatTurnRequest, RouteDecision
from r3.agent.events import DeltaEvent, DoneEvent, ErrorEvent, parse_sse
from r3.agent.service import Advisor, UnconfiguredAdvisor, build_advisor
from r3.agent.tools import AdvisorDeps
from r3.settings import ControllerSettings, ModelChoice
from r3.web import create_app
from r3.web import routes_chat
from tests.fakes import FakeRuns, ScriptedAdvisor
from tests.fakes import test_settings as make_settings
from tests.test_routes import StubDB

MODEL_ID = "openrouter:openrouter/free"


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


def advisor_model(chunks=("There are ", "100 jobs."), *, tool: str | None = "portal_overview", tool_args: str = "{}",
                  seen: list | None = None, route: dict | None = None) -> FunctionModel:
    """Streams one tool call (when ``tool``), then ``chunks`` of text once the
    tool's result is back. ``route`` answers the router's structured call."""

    async def stream(messages, info: AgentInfo):
        if seen is not None:
            seen.append((messages, info))
        last = messages[-1]
        returned = isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts)
        if tool and not returned:
            yield {0: DeltaToolCall(name=tool, json_args=tool_args, tool_call_id="call-1")}
        else:
            for chunk in chunks:
                yield chunk

    def respond(messages, info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, route or {"intent": "advice"})])

    return FunctionModel(respond, stream_function=stream)


def raising(exc: BaseException) -> FunctionModel:
    async def stream(messages, info: AgentInfo):
        raise exc
        yield ""  # pragma: no cover - makes this an async generator

    return FunctionModel(stream_function=stream)


def conversation(*texts: str) -> list[dict[str, str]]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": t} for i, t in enumerate(texts)]


@asynccontextmanager
async def client_for(advisor: Any, *, settings: ControllerSettings | None = None, runs: FakeRuns | None = None,
                     database: Any = None):
    settings = settings or make_settings(router_mode="rules")
    app = create_app(database=database or StubDB(), settings=settings, runs=runs or FakeRuns(), advisor=advisor)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client


async def post_turn(client: AsyncClient, *texts: str, **extra: Any) -> list[Any]:
    response = await client.post("/api/chat/stream", json={"messages": conversation(*texts), **extra})
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


def types(events: list[Any]) -> list[str]:
    return [e.type for e in events]


# --- the turn ------------------------------------------------------------------------------

async def test_turn_streams_route_action_tools_deltas_final_done():
    settings = make_settings(router_mode="rules")
    async with client_for(Advisor(settings, models={"chat": advisor_model()}), settings=settings) as client:
        response = await client.post("/api/chat/stream", json={"messages": conversation("how is the portal doing?")})
    assert response.headers["cache-control"] == "no-cache" and response.headers["x-accel-buffering"] == "no"
    events = parse_sse(response.text)
    assert types(events) == ["route", "action", "tool_call", "tool_result", "delta", "delta", "final", "done"]
    route, action, call, result, *_ = events
    assert route.decision.intent == "advice" and route.decision.router == "rules"
    assert action.action.kind == "advice"
    assert call.name == "portal_overview" and call.args == {}
    assert result.name == "portal_overview" and result.summary.startswith("Jobs stored: 100")
    final = events[-2]
    assert final.text == "There are 100 jobs." == "".join(e.text for e in events if e.type == "delta")
    assert final.model == MODEL_ID and final.usage.requests == 2


async def test_tool_arguments_reach_the_database():
    stub = StubDB()
    model = advisor_model(tool="search_jobs", tool_args='{"query": "kafka", "role": "sre", "remote": true, "limit": 3}')
    settings = make_settings(router_mode="rules")
    async with client_for(Advisor(settings, models={"chat": model}), settings=settings, database=stub) as client:
        events = await post_turn(client, "what do you know about kafka roles?")
    call = next(e for e in events if e.type == "tool_call")
    assert call.args == {"query": "kafka", "role": "sre", "remote": True, "limit": 3}
    f = stub.last_filter
    assert f.search == "kafka" and f.status == "new" and f.remote and stub.last_page == (1, 3)
    result = next(e for e in events if e.type == "tool_result")
    assert result.summary.startswith("101 matching jobs") and len(result.summary) <= 200


async def test_stats_turn_records_the_run_and_narrates_it():
    seen: list = []
    runs = FakeRuns()
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": advisor_model(("Here are the numbers.",), tool=None, seen=seen)})
    async with client_for(advisor, settings=settings, runs=runs) as client:
        events = await post_turn(client, "show me the stats for this week")
    assert types(events) == ["route", "action", "delta", "final", "done"]
    assert events[0].decision.intent == "stats" and events[1].action.kind == "stats"
    assert events[1].action.run is not None
    assert runs.requests[0].params.days == 7 and runs.requests[0].source == "chat"
    instructions = seen[0][1].instructions
    assert "shows a statistics snapshot" in instructions and "the panel next to this chat" in instructions
    assert "never claim you did it" in instructions


async def test_a_job_named_by_id_is_listed_with_its_link():
    seen: list = []
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": advisor_model(("Here it is.",), tool=None, seen=seen)})
    async with client_for(advisor, settings=settings) as client:
        events = await post_turn(client, "tell me about job 7")
    assert events[0].decision.intent == "jobs" and events[1].action.kind == "jobs"
    assert "[Senior ML Engineer at Acme AI](/jobs/7)" in seen[0][1].instructions  # a link, no "#7" (OQ-7)


async def test_history_reaches_the_model():
    seen: list = []
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": advisor_model(("Sure.",), tool=None, seen=seen)})
    async with client_for(advisor, settings=settings) as client:
        await post_turn(client, "hi", "Hello! How can I help?", "what should I focus on?")
    messages = seen[0][0]
    assert isinstance(messages[0], ModelRequest) and messages[0].parts[0].content == "hi"
    assert isinstance(messages[1], ModelResponse) and messages[1].parts[0].content == "Hello! How can I help?"


async def test_advice_mode_skips_the_router():
    runs = FakeRuns()
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": advisor_model(("Ok.",), tool=None)})
    async with client_for(advisor, settings=settings, runs=runs) as client:
        events = await post_turn(client, "show me the stats", mode="advice")
    assert events[0].decision.intent == "advice" and events[0].decision.reason == "advice mode"
    assert events[1].action.kind == "advice" and not runs.requests


@pytest.mark.parametrize("include, needle", [
    (True, "Current summary"), (False, "summary is turned off"),
])
async def test_portal_summary_follows_include_context(include, needle):
    seen: list = []
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": advisor_model(("Ok.",), tool=None, seen=seen)})
    async with client_for(advisor, settings=settings) as client:
        await post_turn(client, "hello there, anything I should know?", include_context=include)
    instructions = seen[0][1].instructions
    assert needle in instructions
    assert ('"total_jobs": 100' in instructions) is include


async def test_llm_router_runs_on_the_chat_model():
    runs = FakeRuns()
    settings = make_settings(router_mode="llm")
    model = advisor_model(("On it.",), tool=None, route={"intent": "stats", "days": 7, "reason": "numbers"})
    async with client_for(Advisor(settings, models={"chat": model}), settings=settings, runs=runs) as client:
        events = await post_turn(client, "how are we doing this week?")
    route = events[0].decision
    assert route.router == "llm" and route.intent == "stats" and route.params.days == 7
    assert events[1].action.kind == "stats" and events[1].action.run is not None


async def test_test_model_answers_with_a_tool_call():
    settings = make_settings(router_mode="rules")
    model = TestModel(call_tools=["pipeline_stats"], custom_output_text="All good.")
    async with client_for(Advisor(settings, models={"chat": model}), settings=settings) as client:
        events = await post_turn(client, "hello there, give me a read on things")
    assert types(events)[:4] == ["route", "action", "tool_call", "tool_result"]
    assert events[2].name == "pipeline_stats"
    assert events[-2].type == "final" and events[-2].text == "All good."
    assert types(events)[-1] == "done"


# --- errors -----------------------------------------------------------------------------------

@pytest.mark.parametrize("model, message", [
    ("foo:bar", "No API key for foo"),
    ("typesafe:jev-latest", "not on the configured list"),
])
async def test_model_resolution_errors_are_not_retryable(model, message):
    settings = make_settings(router_mode="rules")
    async with client_for(Advisor(settings, models={"chat": advisor_model()}), settings=settings) as client:
        events = await post_turn(client, "hi", model=model)
    assert types(events) == ["error", "done"]
    assert message in events[0].message and events[0].retryable is False


async def test_jev_cannot_be_the_chat_model():
    settings = make_settings(router_mode="rules", models=(ModelChoice("typesafe:jev-latest", "Jev"),))
    async with client_for(Advisor(settings, models={"chat": advisor_model()}), settings=settings) as client:
        events = await post_turn(client, "hi", model="typesafe:jev-latest")
    assert types(events) == ["error", "done"] and "cannot chat" in events[0].message


async def test_custom_models_can_be_switched_off():
    settings = make_settings(router_mode="rules", allow_custom_models=False)
    async with client_for(Advisor(settings, models={"chat": advisor_model()}), settings=settings) as client:
        events = await post_turn(client, "hi", model="openai:gpt-5")
    assert types(events) == ["error", "done"] and "not on the configured list" in events[0].message


@pytest.mark.parametrize("exc, message, retryable", [
    (ModelHTTPError(401, "openrouter/free", body="sk-or-secret-body"), "rejected the credentials for OpenRouter", False),
    (ModelHTTPError(429, "openrouter/free", body="sk-or-secret-body"), "usage limit", True),
    (ModelHTTPError(500, "openrouter/free", body="sk-or-secret-body"), "could not answer (HTTP 500)", True),
    (ModelAPIError("openrouter/free", "Request timed out. sk-or-secret-body"), "took too long", True),
    (ModelAPIError("openrouter/free", "connection refused sk-or-secret-body"), "Could not reach OpenRouter", True),
])
async def test_provider_errors_are_safe(exc, message, retryable):
    if "timed out" in str(exc):
        exc.__cause__ = TimeoutError("read timeout")
    settings = make_settings(router_mode="rules")
    async with client_for(Advisor(settings, models={"chat": raising(exc)}), settings=settings) as client:
        response = await client.post("/api/chat/stream", json={"messages": conversation("hello there, status?")})
    events = parse_sse(response.text)
    assert types(events) == ["route", "action", "error", "done"]
    assert message in events[2].message and events[2].retryable is retryable
    assert "secret" not in response.text


def test_error_event_mapping():
    assert "OPENAI_API_KEY" in error_event(ModelHTTPError(403, "gpt-4.1"), "openai:gpt-4.1").message
    assert error_event(ModelHTTPError(404, "x"), "openai:gpt-5").retryable is False
    assert "too many steps" in error_event(UsageLimitExceeded("request_limit"), MODEL_ID).message
    assert "took too long" in error_event(TimeoutError(), MODEL_ID).message
    assert error_event(RuntimeError("boom sk-secret"), MODEL_ID).message == "The assistant could not answer. Please try again."


async def test_the_turn_has_a_deadline():
    async def slow(messages, info: AgentInfo):
        await asyncio.sleep(5)
        yield "late"

    decision = RouteDecision(intent="advice", router="rules")
    deps = AdvisorDeps(database=StubDB(), runs=FakeRuns(), settings=make_settings(), decision=decision,
                       action=AdviceAction(), include_context=False)
    turn = ChatTurnRequest(messages=conversation("hi"))
    events = [e async for e in stream_advice(turn, deps, FunctionModel(stream_function=slow), MODEL_ID, timeout_s=0.2)]
    assert len(events) == 1 and isinstance(events[0], ErrorEvent) and "took too long" in events[0].message


async def test_an_unexpected_failure_still_ends_the_stream():
    class Broken(ScriptedAdvisor):
        async def stream_turn(self, turn, database, runs, *, user=None):
            yield DeltaEvent(text="partial")
            raise RuntimeError("boom")

    async with client_for(Broken(make_settings())) as client:
        events = await post_turn(client, "hi")
    assert types(events) == ["delta", "error", "done"] and "boom" not in events[1].message


async def test_the_stream_stops_at_done():
    advisor = ScriptedAdvisor(make_settings(), [DoneEvent(), DeltaEvent(text="after done")])
    async with client_for(advisor) as client:
        assert types(await post_turn(client, "hi")) == ["done"]


@pytest.mark.parametrize("body", [
    {"messages": []},
    {"messages": [{"role": "assistant", "content": "hi"}]},
    {"messages": [{"role": "user", "content": "   "}]},
    {"messages": [{"role": "user", "content": "hi"}], "mode": "workflow"},
    {"messages": [{"role": "user", "content": "hi"}], "extra": 1},
])
async def test_invalid_bodies_are_422(body):
    async with client_for(ScriptedAdvisor(make_settings())) as client:
        assert (await client.post("/api/chat/stream", json=body)).status_code == 422


async def test_without_a_key_the_chat_explains_the_setup():
    settings = ControllerSettings(api_keys={})
    advisor = build_advisor(settings)
    assert isinstance(advisor, UnconfiguredAdvisor)
    async with client_for(advisor, settings=settings) as client:
        events = await post_turn(client, "hi")
    assert types(events) == ["error", "done"] and "OPENROUTER_API_KEY" in events[0].message
    assert events[0].retryable is False


def test_build_advisor_with_a_key():
    assert isinstance(build_advisor(make_settings()), Advisor)
    assert isinstance(build_advisor(ControllerSettings(api_keys={"typesafe": "ts"})), UnconfiguredAdvisor)


async def test_default_falls_back_to_a_ready_provider():
    seen: list = []
    settings = ControllerSettings(api_keys={"openai": "oa-test"}, router_mode="rules")
    advisor = Advisor(settings, models={"openai:gpt-4.1-mini": advisor_model(("Hi.",), tool=None, seen=seen)})
    async with client_for(advisor, settings=settings) as client:
        events = await post_turn(client, "hi")
    assert events[-2].type == "final" and events[-2].model == "openai:gpt-4.1-mini" and seen


# --- the model picker -----------------------------------------------------------------------------

async def test_models_endpoint_shape():
    async with client_for(ScriptedAdvisor(make_settings()), settings=make_settings()) as client:
        body = (await client.get("/api/models")).json()
    assert body["default"] == MODEL_ID and body["custom_allowed"] is True
    assert body["providers"] == {"openrouter": True, "openai": True, "typesafe": True}
    assert body["models"][:2] == [
        {"id": "openrouter:openai/gpt-6-luna", "label": "OpenRouter · GPT-6 Luna", "provider": "openrouter"},
        {"id": MODEL_ID, "label": "OpenRouter · free router", "provider": "openrouter"}]
    assert {m["provider"] for m in body["models"]} == {"openrouter", "openai"}
    assert "catalogue" not in body


async def test_models_endpoint_lists_only_ready_providers():
    settings = ControllerSettings(api_keys={"openai": "oa-test"}, allow_custom_models=False)
    async with client_for(ScriptedAdvisor(settings), settings=settings) as client:
        body = (await client.get("/api/models?live=1")).json()
    assert body["default"] == "openai:gpt-4.1-mini" and body["custom_allowed"] is False
    assert [m["id"] for m in body["models"]] == ["openai:gpt-4.1-mini", "openai:gpt-4.1"]
    assert body["providers"]["openrouter"] is False and body["catalogue"] == []


async def test_models_endpoint_live_catalogue(monkeypatch):
    calls = 0

    async def fake_catalogue(http=None):
        nonlocal calls
        calls += 1
        return [{"id": "openai/gpt-4.1-mini", "name": "OpenAI: GPT-4.1 Mini", "context_length": 1047576,
                 "prompt_price": 0.4, "completion_price": 1.6}]

    monkeypatch.setattr(routes_chat, "openrouter_catalogue", fake_catalogue)
    async with client_for(ScriptedAdvisor(make_settings()), settings=make_settings()) as client:
        body = (await client.get("/api/models", params={"live": "1"})).json()
    assert calls == 1 and body["catalogue"][0]["id"] == "openai/gpt-4.1-mini"
