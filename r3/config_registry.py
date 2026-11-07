"""Every setting the portal's Settings page manages, declared once.

Each entry names the environment variable :class:`~r3.settings.ControllerSettings`
reads, the group it is shown in, how it is edited and validated, whether it
is a secret (stored encrypted, never sent to the browser) and which
connection test proves it. Only names listed here can be stored as portal
overrides; ``DATABASE_URL`` is deliberately absent because the portal is
connected with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlsplit

from r3.settings import PROVIDERS

Kind = Literal["text", "secret", "select", "bool", "int", "ratio", "url", "pem", "model", "models", "address"]

GroupId = Literal["models", "router", "temporal", "database", "jobs"]


@dataclass(frozen=True)
class SettingSpec:
    name: str
    group: GroupId
    label: str
    help: str
    kind: Kind = "text"
    secret: bool = False
    options: tuple[str, ...] = ()
    default: str = ""  # what ControllerSettings uses when nothing is set (shown as a placeholder)
    test: str = ""  # the connection test this value is checked by
    bounds: tuple[float, float] = (0, 0)  # int kinds

    @property
    def max_len(self) -> int:
        if self.kind == "pem":
            return 20_000
        if self.secret:
            return 8_000  # some API tokens run to a couple of kilobytes
        if self.kind == "models":
            return 4_000
        return 1_000


@dataclass(frozen=True)
class TestSpec:
    id: str
    label: str
    group: GroupId
    needs: tuple[str, ...] = ()  # names that must be set for the test to mean anything
    describes: str = ""


@dataclass(frozen=True)
class GroupSpec:
    id: GroupId
    title: str
    description: str
    tests: tuple[str, ...] = field(default=())  # group-level tests (shown in the card header)


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec("models", "Chat models",
              "Provider keys for the advisor and the model picker. A provider is offered once its key is set."),
    GroupSpec("router", "Workflow detection",
              "How each message is classified before the advisor answers."),
    GroupSpec("temporal", "Temporal",
              "Where runs execute. Without an address, runs execute inside the portal process.",
              tests=("temporal",)),
    GroupSpec("database", "Database",
              "The jobs database. DATABASE_URL is set in the environment only: the portal is connected "
              "with it, so it cannot be changed from here.", tests=("database",)),
    GroupSpec("jobs", "Jobs", "What regular accounts see of a job posting."),
)

SETTINGS: tuple[SettingSpec, ...] = (
    # --- chat models ---
    SettingSpec("OPENROUTER_API_KEY", "models", "OpenRouter API key",
                "Enables the openrouter: models (sk-or-…).", kind="secret", secret=True, test="openrouter"),
    SettingSpec("OPENAI_API_KEY", "models", "OpenAI API key",
                "Enables the openai: models (sk-…).", kind="secret", secret=True, test="openai"),
    SettingSpec("R3_CHAT_MODEL", "models", "Default chat model",
                "provider:name, e.g. openrouter:openai/gpt-6-luna or openai:gpt-4.1-mini.",
                kind="model", default="openrouter:openai/gpt-6-luna"),
    SettingSpec("R3_CHAT_MODELS", "models", "Model picker",
                "Comma-separated id or id=Label entries. Empty means the built-in list.", kind="models"),
    SettingSpec("R3_TASK_MODEL", "models", "AI steps model",
                "Model for the resume builder, the resume reader and job overviews, apart from the "
                "chat. Empty means the default chat model.", kind="model"),
    SettingSpec("R3_CHAT_ALLOW_CUSTOM_MODELS", "models", "Allow custom model ids",
                "Let the picker take any openrouter:/openai: id, not only the listed ones.",
                kind="bool", default="1"),
    # --- workflow detection ---
    SettingSpec("R3_ROUTER_MODE", "router", "Router mode",
                "rules: keywords only · llm: the chat model · jev: TypeSafe's Jev · hybrid: Jev, then the "
                "LLM below the confidence floor.", kind="select", options=("rules", "llm", "jev", "hybrid"),
                default="llm"),
    SettingSpec("R3_ROUTER_MODEL", "router", "Router model",
                "Model for the llm router. Empty means the model chosen for the turn.", kind="model"),
    SettingSpec("TYPESAFE_API_KEY", "router", "TypeSafe API key",
                "Enables the jev and hybrid routers.", kind="secret", secret=True, test="typesafe"),
    SettingSpec("R3_TYPESAFE_MODEL", "router", "Jev model",
                "typesafe:name.", kind="model", default="typesafe:jev-latest"),
    SettingSpec("R3_JEV_CONFIDENCE", "router", "Jev confidence floor",
                "Between 0 and 1. In hybrid mode, below this the LLM decides.", kind="ratio", default="0.6"),
    # --- temporal ---
    SettingSpec("TEMPORAL_ADDRESS", "temporal", "Address",
                "host:port, e.g. <namespace>.<account>.tmprl.cloud:7233 or localhost:7233.", kind="address"),
    SettingSpec("TEMPORAL_NAMESPACE", "temporal", "Namespace", "e.g. r3.<account>.", default="default"),
    SettingSpec("TEMPORAL_API_KEY", "temporal", "API key",
                "Temporal Cloud API key. Leave empty for mTLS or a local server.", kind="secret", secret=True),
    SettingSpec("TEMPORAL_TLS_CERT", "temporal", "mTLS client certificate",
                "PEM text, or a path on the server. Used when no API key is set.", kind="pem"),
    SettingSpec("TEMPORAL_TLS_KEY", "temporal", "mTLS client key",
                "PEM text, or a path on the server.", kind="pem", secret=True),
    SettingSpec("TEMPORAL_TASK_QUEUE", "temporal", "Task queue", "The queue R3's worker polls.",
                default="r3-controller"),
    SettingSpec("TEMPORAL_UI_URL", "temporal", "Temporal UI URL",
                "Base URL listing workflows, for \"Open in Temporal\" links. Derived for tmprl.cloud and localhost.",
                kind="url"),
    SettingSpec("R3_RUNS_BACKEND", "temporal", "Runs backend",
                "auto: Temporal when an address is set, else in-process · temporal · local · none.",
                kind="select", options=("auto", "temporal", "local", "none"), default="auto"),
    SettingSpec("R3_EMBED_WORKER", "temporal", "Embedded worker",
                "Run the Temporal worker inside the portal. Off means a separate r3-worker polls.",
                kind="bool", default="1"),
    # --- jobs ---
    SettingSpec("R3_JOB_POSTINGS", "jobs", "Job postings for regular accounts",
                "full: every account sees the posting, its description and its links · teaser: only the "
                "role, its stack, pay and an overview (admins always see the posting).",
                kind="select", options=("full", "teaser"), default="full"),
)

TESTS: tuple[TestSpec, ...] = (
    TestSpec("openrouter", "OpenRouter", "models", ("OPENROUTER_API_KEY",), "GET openrouter.ai/api/v1/auth/key"),
    TestSpec("openai", "OpenAI", "models", ("OPENAI_API_KEY",), "GET api.openai.com/v1/models"),
    TestSpec("typesafe", "TypeSafe", "router", ("TYPESAFE_API_KEY",), "one Noul question to Jev"),
    TestSpec("temporal", "Temporal", "temporal", ("TEMPORAL_ADDRESS",),
             "connect, describe the namespace, look for a worker on the task queue"),
    TestSpec("database", "Database", "database", (), "SELECT 1 and the overview counts"),
)

BY_NAME: dict[str, SettingSpec] = {s.name: s for s in SETTINGS}
TESTS_BY_ID: dict[str, TestSpec] = {t.id: t for t in TESTS}
GROUPS_BY_ID: dict[str, GroupSpec] = {g.id: g for g in GROUPS}
SECRET_NAMES: frozenset[str] = frozenset(s.name for s in SETTINGS if s.secret)

# Groups whose values the run service (and its worker) is built from; the
# rest only feed the advisor and the settings object.
RUNS_GROUPS: frozenset[str] = frozenset({"temporal"})
MODEL_GROUPS: frozenset[str] = frozenset({"models", "router"})

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_ANY_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MODEL = re.compile(r"^[a-z]+:[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}$")
_HOSTPORT = re.compile(r"^(\[[0-9a-fA-F:]+\]|[A-Za-z0-9.-]+):(\d{1,5})$")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def spec(name: str) -> SettingSpec:
    try:
        return BY_NAME[name]
    except KeyError:
        raise ValueError(f"{name} is not a setting the portal manages") from None


def settings_in(group: str) -> list[SettingSpec]:
    return [s for s in SETTINGS if s.group == group]


def tests_for(group: str) -> list[TestSpec]:
    return [t for t in TESTS if t.group == group]


def validate(name: str, raw: object) -> str:
    """The value to store for ``name``, normalised, or ``ValueError`` with a
    message the page can show next to the field."""
    s = spec(name)
    if not isinstance(raw, str):
        raise ValueError("must be a string")
    value = raw.replace("\r\n", "\n").strip() if s.kind == "pem" else raw.strip()
    if len(value) > s.max_len:
        raise ValueError(f"is too long (at most {s.max_len} characters)")
    if s.kind == "pem":
        if _CONTROL.search(value):
            raise ValueError("contains control characters")
        if value and not value.startswith("-----BEGIN") and "\n" in value:
            raise ValueError("must be PEM text (-----BEGIN …) or a single file path")
    elif _ANY_CONTROL.search(value):
        raise ValueError("must be a single line without control characters")
    if value == "":
        return ""
    kind = s.kind
    if kind == "secret":
        if "  " in value:
            raise ValueError("contains repeated spaces; paste the key exactly")
        return value
    if kind == "select":
        low = value.lower()
        if low not in s.options:
            raise ValueError(f"must be one of {', '.join(s.options)}")
        return low
    if kind == "bool":
        low = value.lower()
        if low in _TRUE:
            return "1"
        if low in _FALSE:
            return "0"
        raise ValueError("must be on or off")
    if kind == "int":
        try:
            n = int(value)
        except ValueError:
            raise ValueError("must be a whole number") from None
        lo, hi = s.bounds
        if not lo <= n <= hi:
            raise ValueError(f"must be between {int(lo)} and {int(hi)}")
        return str(n)
    if kind == "ratio":
        try:
            x = float(value)
        except ValueError:
            raise ValueError("must be a number between 0 and 1") from None
        if not 0 <= x <= 1:
            raise ValueError("must be between 0 and 1")
        return value
    if kind == "url":
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("must be an http(s):// URL")
        return value.rstrip("/")
    if kind == "address":
        if not _HOSTPORT.match(value) or not 0 < int(value.rsplit(":", 1)[1]) < 65536:
            raise ValueError("must be host:port, e.g. localhost:7233")
        return value
    if kind == "model":
        _check_model(name, value)
        return value
    if kind == "models":
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            return ""
        for item in items:
            model_id, _, _label = item.partition("=")
            _check_model(name, model_id.strip())
        return ",".join(items)
    if " " in value and name in ("TEMPORAL_NAMESPACE", "TEMPORAL_TASK_QUEUE"):
        raise ValueError("must not contain spaces")
    return value


def _check_model(name: str, model_id: str) -> None:
    if not _MODEL.match(model_id):
        raise ValueError(f"{model_id!r} must look like provider:name")
    provider = model_id.split(":", 1)[0]
    if name == "R3_TYPESAFE_MODEL":
        if provider != "typesafe":
            raise ValueError("must be a typesafe: model")
    elif provider not in PROVIDERS or provider == "typesafe":
        raise ValueError(f"provider must be openrouter or openai, not {provider!r}")
