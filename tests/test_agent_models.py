"""The model registry: ids to providers, the per-(id, key) cache, the
OpenRouter catalogue (mocked transport), and chat history conversion."""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.typesafe import TypeSafeModel

from r3.agent import models as registry
from r3.agent.contracts import ChatMessage
from r3.agent.models import build_model, history_messages, openrouter_catalogue, split_model_id
from tests.fakes import test_settings as make_settings


@pytest.fixture(autouse=True)
def _offline():
    registry.clear_model_cache()
    registry.clear_catalogue_cache()
    with override_allow_model_requests(False):
        yield
    registry.clear_model_cache()
    registry.clear_catalogue_cache()


@pytest.mark.parametrize("model_id, cls, name", [
    ("openrouter:openai/gpt-4.1-mini", OpenRouterModel, "openai/gpt-4.1-mini"),
    ("openrouter:openrouter/free", OpenRouterModel, "openrouter/free"),
    ("openai:gpt-4.1-mini", OpenAIChatModel, "gpt-4.1-mini"),
    ("typesafe:jev-latest", TypeSafeModel, "jev-latest"),
])
def test_build_model_per_provider(model_id, cls, name):
    model = build_model(model_id, make_settings())
    assert isinstance(model, cls)
    assert model.model_name == name


def test_openrouter_model_carries_the_app_title_and_key():
    model = build_model("openrouter:openai/gpt-4.1-mini", make_settings())
    client = model.client
    assert client.api_key == "or-test"
    assert client.base_url.host == "openrouter.ai"
    assert client.default_headers.get("X-Title") == "R3 Controller"


def test_openai_model_uses_the_settings_key_not_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    model = build_model("openai:gpt-4.1", make_settings())
    assert model.client.api_key == "oa-test"
    assert model.client.base_url.host == "api.openai.com"


def test_models_are_cached_per_id_and_key():
    settings = make_settings()
    first = build_model("openai:gpt-4.1-mini", settings)
    assert build_model("openai:gpt-4.1-mini", settings) is first
    assert build_model("openai:gpt-4.1", settings) is not first
    rotated = make_settings(api_keys={"openai": "oa-rotated"})
    again = build_model("openai:gpt-4.1-mini", rotated)
    assert again is not first and again.client.api_key == "oa-rotated"
    assert all("oa-test" not in str(key) for key in registry._MODEL_CACHE)  # only a digest of the key is kept


@pytest.mark.parametrize("model_id, message", [
    ("anthropic:claude-sonnet-4", "Unknown model provider"),
    ("gpt-4.1-mini", "provider:name"),
    ("openai:", "provider:name"),
])
def test_build_model_rejects_unknown_ids(model_id, message):
    with pytest.raises(ValueError, match=message):
        build_model(model_id, make_settings())


def test_build_model_needs_the_provider_key():
    settings = make_settings(api_keys={"openrouter": "or-test"})
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_model("openai:gpt-4.1-mini", settings)
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        build_model("typesafe:jev-latest", settings)


def test_split_model_id():
    assert split_model_id(" openrouter:anthropic/claude-sonnet-4 ") == ("openrouter", "anthropic/claude-sonnet-4")


# --- the OpenRouter catalogue -------------------------------------------------------------

CATALOGUE = {"data": [
    {"id": "openai/gpt-4.1-mini", "name": "OpenAI: GPT-4.1 Mini", "context_length": 1047576,
     "pricing": {"prompt": "0.0000004", "completion": "0.0000016"}},
    {"id": "anthropic/claude-sonnet-4", "name": "Anthropic: Claude Sonnet 4", "context_length": 200000,
     "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "openrouter/auto", "name": "Auto Router", "context_length": 2000000,
     "pricing": {"prompt": "-1", "completion": "-1"}},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta: Llama 3.3 70B (free)",
     "pricing": {"prompt": "0", "completion": "0"}},
    {"name": "no id"},
]}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_catalogue_is_parsed_and_sorted_by_name():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=CATALOGUE)

    async with _client(handler) as http:
        entries = await openrouter_catalogue(http)
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/models"
    assert "authorization" not in seen[0].headers  # public endpoint: no key is sent
    assert [e["id"] for e in entries] == ["anthropic/claude-sonnet-4", "openrouter/auto",
                                          "meta-llama/llama-3.3-70b-instruct:free", "openai/gpt-4.1-mini"]
    gpt = entries[-1]
    assert gpt == {"id": "openai/gpt-4.1-mini", "name": "OpenAI: GPT-4.1 Mini", "context_length": 1047576,
                   "prompt_price": 0.4, "completion_price": 1.6}
    assert entries[1]["prompt_price"] is None  # a router's price depends on the model it picks
    assert entries[2]["prompt_price"] == 0.0 and entries[2]["context_length"] is None


async def test_catalogue_is_cached_for_ten_minutes():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=CATALOGUE)

    async with _client(handler) as http:
        first = await openrouter_catalogue(http)
        second = await openrouter_catalogue(http)
    assert calls == 1 and first == second


@pytest.mark.parametrize("response", [
    httpx.Response(503, text="upstream down"),
    httpx.Response(200, text="not json"),
    httpx.Response(200, json={"unexpected": True}),
])
async def test_catalogue_failures_are_empty(response):
    async with _client(lambda request: response) as http:
        assert await openrouter_catalogue(http) == []


async def test_catalogue_network_error_is_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route")

    async with _client(handler) as http:
        assert await openrouter_catalogue(http) == []


def test_history_messages_alternate_requests_and_responses():
    history = [ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="x" * 50)]
    messages = history_messages(history, max_chars=10)
    assert isinstance(messages[0], ModelRequest) and messages[0].parts[0].content == "hi"
    assert isinstance(messages[1], ModelResponse) and messages[1].parts[0].content == "x" * 9 + "…"


def test_default_chat_model_is_gpt_6_luna_everywhere():
    """The Assistant defaults to a fast paid model, not the free router: the
    settings default, the picker and the Settings page placeholder agree."""
    from r3.config_registry import spec
    from r3.settings import DEFAULT_MODELS, ControllerSettings

    luna = "openrouter:openai/gpt-6-luna"
    assert ControllerSettings().default_model == luna
    assert ControllerSettings.from_env({}).default_model == luna
    assert DEFAULT_MODELS[0].id == luna and DEFAULT_MODELS[0].label == "OpenRouter · GPT-6 Luna"
    assert "openrouter:openrouter/free" in [m.id for m in DEFAULT_MODELS]
    assert spec("R3_CHAT_MODEL").default == luna
