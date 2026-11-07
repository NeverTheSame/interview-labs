"""The model registry: turn a ``provider:name`` id into a Pydantic AI model,
and read OpenRouter's public catalogue for the picker.

Providers are built with the key :class:`~r3.settings.ControllerSettings`
already read from the environment; nothing here reads a key from the
environment itself. Built models are cached per (id, key) so an agent does
not construct a provider and an HTTP client on every request.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any, Sequence

import httpx
import pydantic_ai
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model

from r3.settings import PROVIDER_KEYS, PROVIDERS, ControllerSettings

if TYPE_CHECKING:
    from .contracts import ChatMessage

log = logging.getLogger("r3.agent.models")

# The portal owns its output: no first-run banner in the server log.
pydantic_ai.BANNER_ENABLED = False

APP_TITLE = "R3 Controller"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

_MODEL_CACHE: dict[tuple[str, str], Model] = {}
_MODEL_CACHE_MAX = 32


def split_model_id(model_id: str) -> tuple[str, str]:
    """``"openrouter:openai/gpt-4.1-mini"`` → ``("openrouter", "openai/gpt-4.1-mini")``.
    Raises ``ValueError`` for an id without a provider or a name."""
    provider, sep, name = model_id.strip().partition(":")
    if not sep or not provider or not name.strip():
        raise ValueError(f"Model id {model_id!r} must look like provider:name")
    return provider, name.strip()


def build_model(model_id: str, settings: ControllerSettings) -> Model:
    """The Pydantic AI model for ``model_id`` (``openrouter:<vendor/model>``,
    ``openai:<model>`` or ``typesafe:<jev model>``), using the provider's key
    from ``settings``. Raises ``ValueError`` for an unknown provider or a
    provider without a key."""
    provider, name = split_model_id(model_id)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown model provider {provider!r}: use one of {', '.join(PROVIDERS)}")
    key = settings.key_for(provider)
    if not key:
        raise ValueError(f"No API key for {provider}: set {PROVIDER_KEYS[provider]}")
    # The key is part of the cache identity (a rotated key gets a new client),
    # but only its digest is kept, so the cache never holds a second copy.
    cache_key = (f"{provider}:{name}", hashlib.sha256(key.encode()).hexdigest())
    if (model := _MODEL_CACHE.get(cache_key)) is not None:
        return model
    model = _construct(provider, name, key)
    if len(_MODEL_CACHE) >= _MODEL_CACHE_MAX:
        _MODEL_CACHE.pop(next(iter(_MODEL_CACHE)))
    _MODEL_CACHE[cache_key] = model
    return model


def _construct(provider: str, name: str, key: str) -> Model:
    # Imported per provider: each pulls in its SDK, and a portal that only
    # talks to OpenRouter never needs TypeSafe's.
    if provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel
        from pydantic_ai.providers.openrouter import OpenRouterProvider

        return OpenRouterModel(name, provider=OpenRouterProvider(api_key=key, app_title=APP_TITLE))
    if provider == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(name, provider=OpenAIProvider(base_url=OPENAI_BASE_URL, api_key=key))
    if provider == "typesafe":
        from pydantic_ai.models.typesafe import TypeSafeModel
        from pydantic_ai.providers.typesafe import TypeSafeProvider

        return TypeSafeModel(name, provider=TypeSafeProvider(api_key=key))
    raise ValueError(f"Unknown model provider {provider!r}")  # pragma: no cover - checked by the caller


def clear_model_cache() -> None:
    """Forget every built model (tests, key rotation)."""
    _MODEL_CACHE.clear()


def history_messages(history: Sequence[ChatMessage], *, max_chars: int | None = None) -> list[ModelMessage]:
    """A chat transcript as Pydantic AI message history: user turns become
    requests, assistant turns responses. ``max_chars`` trims long turns."""
    out: list[ModelMessage] = []
    for message in history:
        content = message.content
        if max_chars is not None and len(content) > max_chars:
            content = content[: max_chars - 1] + "…"
        if message.role == "user":
            out.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        else:
            out.append(ModelResponse(parts=[TextPart(content=content)]))
    return out


# --- OpenRouter's catalogue -------------------------------------------------------

CATALOGUE_TTL_S = 600.0
CATALOGUE_FAILURE_TTL_S = 60.0
CATALOGUE_TIMEOUT_S = 5.0

_catalogue: tuple[float, list[dict[str, Any]]] | None = None  # (expires at, entries)


async def openrouter_catalogue(http: httpx.AsyncClient | None = None) -> list[dict[str, Any]]:
    """OpenRouter's public model list (no key needed), for the picker's
    "custom model" search: ``[{id, name, context_length, prompt_price,
    completion_price}]`` sorted by name. Prices are USD per million tokens
    (``None`` when OpenRouter does not state one, ``0.0`` for free models).

    Cached for 10 minutes. On any failure the result is ``[]`` (remembered
    for a minute, so a down endpoint is not hammered) and callers fall back
    to the configured list."""
    global _catalogue
    now = time.monotonic()
    if _catalogue is not None and _catalogue[0] > now:
        return _catalogue[1]
    try:
        if http is None:
            async with httpx.AsyncClient(timeout=CATALOGUE_TIMEOUT_S) as client:
                response = await client.get(OPENROUTER_MODELS_URL)
        else:
            response = await http.get(OPENROUTER_MODELS_URL, timeout=CATALOGUE_TIMEOUT_S)
        response.raise_for_status()
        entries = _parse_catalogue(response.json())
    except Exception as exc:  # network, status, JSON shape: the picker just goes without
        log.warning("OpenRouter catalogue unavailable (%s)", type(exc).__name__)
        _catalogue = (now + CATALOGUE_FAILURE_TTL_S, [])
        return []
    _catalogue = (now + CATALOGUE_TTL_S, entries)
    return entries


def clear_catalogue_cache() -> None:
    global _catalogue
    _catalogue = None


def _parse_catalogue(payload: Any) -> list[dict[str, Any]]:
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("unexpected catalogue shape")
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
            continue
        pricing = row.get("pricing") if isinstance(row.get("pricing"), dict) else {}
        context = row.get("context_length")
        out.append({
            "id": row["id"].strip(),
            "name": str(row.get("name") or row["id"]).strip(),
            "context_length": context if isinstance(context, int) and not isinstance(context, bool) else None,
            "prompt_price": _per_million(pricing.get("prompt")),
            "completion_price": _per_million(pricing.get("completion")),
        })
    out.sort(key=lambda entry: (entry["name"].lower(), entry["id"]))
    return out


def _per_million(raw: Any) -> float | None:
    """OpenRouter prices are USD per token, as strings; ``-1`` marks a
    router whose price depends on the model it picks."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return round(value * 1_000_000, 6)
