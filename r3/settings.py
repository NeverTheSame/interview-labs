"""Controller settings: which models the chat may use, how a message is
routed to a workflow and where Temporal is. Everything is read from the
environment once at startup.

Model ids are ``provider:name``: ``openrouter:openai/gpt-4.1-mini``,
``openai:gpt-4.1-mini``, ``typesafe:jev-latest``. The provider names the API
key that must be present (``OPENROUTER_API_KEY``, ``OPENAI_API_KEY``,
``TYPESAFE_API_KEY``); a model whose provider has no key is not offered.
The pre-agent variables ``R3_CHAT_MODEL`` (a bare name) and
``R3_CHAT_BASE_URL`` are still honoured: a bare name is attributed to
OpenRouter when the base URL points at openrouter.ai and to OpenAI otherwise.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal, Mapping
from urllib.parse import urlsplit

Provider = Literal["openrouter", "openai", "typesafe"]
PROVIDERS: tuple[Provider, ...] = ("openrouter", "openai", "typesafe")
PROVIDER_KEYS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
}

RouterMode = Literal["rules", "llm", "jev", "hybrid"]
RunsBackend = Literal["auto", "temporal", "local", "none"]
JobPostings = Literal["full", "teaser"]

# A model id the operator typed rather than picked: provider, colon, a vendor/model path.
_MODEL_ID = re.compile(r"^(openrouter|openai):[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}$")


@dataclass(frozen=True)
class ModelChoice:
    """One entry of the model picker."""

    id: str
    label: str

    @property
    def provider(self) -> str:
        return self.id.split(":", 1)[0]

    @property
    def name(self) -> str:
        return self.id.split(":", 1)[1]


DEFAULT_MODELS: tuple[ModelChoice, ...] = (
    ModelChoice("openrouter:openai/gpt-6-luna", "OpenRouter · GPT-6 Luna"),
    ModelChoice("openrouter:openrouter/free", "OpenRouter · free router"),
    ModelChoice("openrouter:openai/gpt-4.1-mini", "OpenRouter · GPT-4.1 mini"),
    ModelChoice("openrouter:anthropic/claude-sonnet-4", "OpenRouter · Claude Sonnet 4"),
    ModelChoice("openrouter:google/gemini-2.5-flash", "OpenRouter · Gemini 2.5 Flash"),
    ModelChoice("openai:gpt-4.1-mini", "OpenAI · GPT-4.1 mini"),
    ModelChoice("openai:gpt-4.1", "OpenAI · GPT-4.1"),
)


def parse_model_id(raw: str, base_url: str = "") -> str:
    """Normalise a model id to ``provider:name``. A bare name is attributed by
    the chat base URL, which is how the pre-agent configuration named models."""
    raw = raw.strip()
    if not raw:
        return ""
    if raw.split(":", 1)[0] in PROVIDERS and ":" in raw:
        return raw
    host = urlsplit(base_url).hostname or ""
    provider = "openrouter" if host.endswith("openrouter.ai") else "openai"
    return f"{provider}:{raw}"


def _parse_models(spec: str, base_url: str) -> tuple[ModelChoice, ...]:
    """``R3_CHAT_MODELS`` is a comma-separated list of ``id`` or ``id=Label``."""
    out: list[ModelChoice] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        model_id, _, label = item.partition("=")
        model_id = parse_model_id(model_id, base_url)
        if model_id:
            out.append(ModelChoice(model_id, label.strip() or model_id))
    return tuple(out)


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ControllerSettings:
    api_keys: Mapping[str, str] = field(default_factory=dict, repr=False)
    default_model: str = "openrouter:openai/gpt-6-luna"
    models: tuple[ModelChoice, ...] = DEFAULT_MODELS
    allow_custom_models: bool = True

    # How a message is classified before the advisor answers. "rules" is
    # keyword-only (no model), "llm" asks the chat model for a structured
    # route, "jev" asks TypeSafe's Jev, "hybrid" asks Jev and falls back to
    # the LLM below ``jev_confidence`` or when Jev cannot fill the fields.
    router_mode: RouterMode = "llm"
    router_model: str = ""  # "" means the chat model chosen for the turn
    # The model that reads an uploaded resume (r3/resume/parse.py); "" means
    # the chat's effective default. The task is line labeling: a free model.
    parse_model: str = ""
    # The model for R3's AI steps (the resume builder and reader, job
    # overviews); "" means the chat's effective default. Lets the chat run
    # on a free model while the steps that must not fail run on a stronger one.
    task_model: str = ""
    typesafe_model: str = "typesafe:jev-latest"
    jev_confidence: float = 0.6

    temporal_address: str = ""
    temporal_namespace: str = "default"
    temporal_api_key: str = field(default="", repr=False)
    temporal_tls_cert: str = ""  # PEM paths, for mTLS namespaces
    temporal_tls_key: str = ""
    temporal_task_queue: str = "r3-controller"
    temporal_ui_url: str = ""  # base URL that lists workflows; derived when empty
    runs_backend: RunsBackend = "auto"

    # Who sees a job's full posting: "full" shows every account the posting,
    # its description and its links (while the owner dogfoods R3); "teaser"
    # gives a regular account the OQ-6 teaser instead. Admins always see it.
    job_postings: JobPostings = "full"

    # Run the Temporal worker inside the portal process, so one process
    # serves the page and executes the runs. Turn it off to run
    # ``r3-worker`` as its own process.
    embed_worker: bool = True

    include_context: bool = True

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ControllerSettings":
        env = os.environ if env is None else env
        base_url = env.get("R3_CHAT_BASE_URL", "").strip()
        default_model = parse_model_id(env.get("R3_CHAT_MODEL", ""), base_url) or cls.default_model
        models = _parse_models(env.get("R3_CHAT_MODELS", ""), base_url) or DEFAULT_MODELS
        if all(m.id != default_model for m in models):
            models = (ModelChoice(default_model, default_model), *models)
        address = env.get("TEMPORAL_ADDRESS", "").strip()
        namespace = env.get("TEMPORAL_NAMESPACE", "").strip() or "default"
        return cls(
            api_keys={p: env.get(PROVIDER_KEYS[p], "").strip() for p in PROVIDERS},
            default_model=default_model,
            models=models,
            allow_custom_models=_flag(env, "R3_CHAT_ALLOW_CUSTOM_MODELS", True),
            router_mode=_router_mode(env.get("R3_ROUTER_MODE", "")),
            router_model=parse_model_id(env.get("R3_ROUTER_MODEL", ""), base_url),
            parse_model=parse_model_id(env.get("R3_PARSE_MODEL", ""), base_url),
            task_model=parse_model_id(env.get("R3_TASK_MODEL", ""), base_url),
            typesafe_model=env.get("R3_TYPESAFE_MODEL", "").strip() or "typesafe:jev-latest",
            jev_confidence=_ratio(env.get("R3_JEV_CONFIDENCE", ""), 0.6),
            temporal_address=address,
            temporal_namespace=namespace,
            temporal_api_key=env.get("TEMPORAL_API_KEY", "").strip(),
            temporal_tls_cert=env.get("TEMPORAL_TLS_CERT", "").strip(),
            temporal_tls_key=env.get("TEMPORAL_TLS_KEY", "").strip(),
            temporal_task_queue=env.get("TEMPORAL_TASK_QUEUE", "").strip() or "r3-controller",
            temporal_ui_url=env.get("TEMPORAL_UI_URL", "").strip() or _derive_ui_url(address, namespace),
            runs_backend=_runs_backend(env.get("R3_RUNS_BACKEND", "")),
            job_postings="teaser" if env.get("R3_JOB_POSTINGS", "").strip().lower() == "teaser" else "full",
            embed_worker=_flag(env, "R3_EMBED_WORKER", True),
            include_context=_flag(env, "R3_CHAT_INCLUDE_CONTEXT", True),
        )

    # --- models ---------------------------------------------------------------

    def key_for(self, provider: str) -> str:
        return self.api_keys.get(provider, "")

    def provider_ready(self, provider: str) -> bool:
        return bool(self.key_for(provider))

    @property
    def chat_ready(self) -> bool:
        return self.provider_ready(self.default_model.split(":", 1)[0])

    @property
    def jev_ready(self) -> bool:
        return self.provider_ready("typesafe")

    def postings_for(self, *, is_admin: bool) -> bool:
        """Whether this viewer sees a job's full posting and its links."""
        return is_admin or self.job_postings == "full"

    def available_models(self) -> list[ModelChoice]:
        """The picker: configured models whose provider has a key."""
        return [m for m in self.models if self.provider_ready(m.provider)]

    def resolve_model(self, requested: str | None) -> str:
        """The model a turn runs on: the default, or a requested id that is
        either on the list or a well-formed custom id for a provider with a
        key. Anything else is refused so a client cannot pick a provider the
        operator never configured."""
        model_id = (requested or "").strip() or self.default_model
        provider = model_id.split(":", 1)[0]
        if not self.provider_ready(provider):
            raise ValueError(f"No API key for {provider}: set {PROVIDER_KEYS.get(provider, 'the provider key')}")
        if any(m.id == model_id for m in self.models):
            return model_id
        if self.allow_custom_models and _MODEL_ID.match(model_id):
            return model_id
        raise ValueError(f"Model {model_id!r} is not on the configured list")

    def setup_hint(self) -> str:
        provider = self.default_model.split(":", 1)[0]
        return (f"Set {PROVIDER_KEYS.get(provider, 'an API key')} in .env and restart the portal to enable chat. "
                "Choose the default with R3_CHAT_MODEL and the picker with R3_CHAT_MODELS.")

    # --- temporal -------------------------------------------------------------

    @property
    def temporal_configured(self) -> bool:
        return bool(self.temporal_address)

    def run_url(self, run_id: str) -> str:
        """A link to one workflow in the Temporal UI, or empty when unknown."""
        return f"{self.temporal_ui_url}/{run_id}" if self.temporal_ui_url else ""


def _router_mode(raw: str) -> RouterMode:
    raw = raw.strip().lower()
    return raw if raw in ("rules", "llm", "jev", "hybrid") else "llm"  # type: ignore[return-value]


def _runs_backend(raw: str) -> RunsBackend:
    raw = raw.strip().lower()
    return raw if raw in ("auto", "temporal", "local", "none") else "auto"  # type: ignore[return-value]


def _ratio(raw: str, default: float) -> float:
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if 0 <= value <= 1 else default


def _derive_ui_url(address: str, namespace: str) -> str:
    """Temporal Cloud and the local dev server have predictable UIs."""
    host = address.split(":", 1)[0]
    if host.endswith(".tmprl.cloud"):
        return f"https://cloud.temporal.io/namespaces/{namespace}/workflows"
    if host in ("localhost", "127.0.0.1", "::1"):
        return f"http://localhost:8233/namespaces/{namespace}/workflows"
    return ""
