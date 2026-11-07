"""The Settings page's "Test connection" checks. Each one calls only its
vendor's named endpoint with the candidate settings and returns a
:class:`CheckResult`; none raises, and no message or detail ever carries a
secret (every secret in play is scrubbed from the output as a last guard).

HTTP checks take an optional ``transport`` (tests pass
``httpx.MockTransport``); the TypeSafe check takes an ``httpx2`` transport,
since that SDK is built on httpx2; the Temporal check takes a ``connect``
coroutine in place of :func:`r3.runs.temporal.connect_client`.
"""

from __future__ import annotations

import asyncio
import socket
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Awaitable, Callable

import httpx

from r3.settings import ControllerSettings

TIMEOUT_S = 10.0
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/auth/key"
OPENAI_MODELS_URL = "https://api.openai.com/v1/models"


@dataclass
class CheckResult:
    ok: bool
    message: str
    latency_ms: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    configured: bool = True

    def as_dict(self, test_id: str) -> dict[str, Any]:
        return {"id": test_id, "ok": self.ok, "message": self.message, "latency_ms": self.latency_ms,
                "details": self.details, "configured": self.configured}


@dataclass
class CheckContext:
    settings: ControllerSettings
    database: Any = None
    transport: httpx.AsyncBaseTransport | None = None
    typesafe_transport: Any = None  # httpx2.AsyncBaseTransport
    connect: Callable[..., Awaitable[Any]] | None = None


class _Clock:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    @property
    def ms(self) -> int:
        return int((time.perf_counter() - self.start) * 1000)


def _client(ctx: CheckContext) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S), transport=ctx.transport)


def _is_dns(exc: BaseException) -> bool:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, socket.gaierror):
            return True
        text = str(exc).lower()
        if any(s in text for s in ("name or service not known", "nodename nor servname", "name resolution",
                                   "getaddrinfo failed", "no address associated")):
            return True
        exc = exc.__cause__ or exc.__context__  # type: ignore[assignment]
    return False


def transport_failure(exc: BaseException, what: str, host: str) -> str:
    """A readable message for a request that got no HTTP answer."""
    if isinstance(exc, (httpx.TimeoutException, TimeoutError, asyncio.TimeoutError)):
        return f"{what} did not answer within {int(TIMEOUT_S)} s."
    if _is_dns(exc):
        return f"Could not resolve {host}: check the address and the server's DNS."
    if isinstance(exc, httpx.ProxyError):
        return f"The server's outbound proxy could not reach {host}."
    if isinstance(exc, httpx.ConnectError):
        return f"Could not connect to {host}."
    return f"{what} is unreachable ({type(exc).__name__})."


def status_failure(code: int, what: str, key_name: str) -> str:
    if code in (401, 403):
        return f"{what} rejected {key_name} (HTTP {code})."
    if code == 404:
        return f"{what} answered 404: the endpoint or resource was not found."
    if code == 429:
        return f"{what} is rate limiting this key (HTTP 429); try again shortly."
    if code >= 500:
        return f"{what} had a server error (HTTP {code}); try again shortly."
    return f"{what} refused the request (HTTP {code})."


async def _get(ctx: CheckContext, what: str, url: str, headers: dict[str, str]) -> tuple[httpx.Response | None, str, int]:
    clock = _Clock()
    host = httpx.URL(url).host
    try:
        async with _client(ctx) as client:
            resp = await client.get(url, headers=headers)
    except Exception as e:  # network errors of every kind become a message
        return None, transport_failure(e, what, host), clock.ms
    return resp, "", clock.ms


def _json(resp: httpx.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


# --- model providers ----------------------------------------------------------------


async def check_openrouter(ctx: CheckContext) -> CheckResult:
    key = ctx.settings.key_for("openrouter")
    if not key:
        return CheckResult(False, "OPENROUTER_API_KEY is not set.", configured=False)
    resp, err, ms = await _get(ctx, "OpenRouter", OPENROUTER_KEY_URL, {"Authorization": f"Bearer {key}"})
    if resp is None:
        return CheckResult(False, err, ms)
    if resp.status_code != 200:
        return CheckResult(False, status_failure(resp.status_code, "OpenRouter", "OPENROUTER_API_KEY"), ms)
    data = (_json(resp) or {}).get("data") or {}
    details: dict[str, Any] = {}
    label = str(data.get("label") or "")
    # Unnamed keys are labelled with a truncated copy of the key itself.
    if label and not label.startswith("sk-"):
        details["label"] = label[:80]
    for k in ("limit", "limit_remaining", "usage", "is_free_tier"):
        if k in data:
            details[k] = data[k]
    limit = data.get("limit")
    parts = ["OpenRouter accepted the key"]
    if details.get("label"):
        parts.append(f"label {details['label']}")
    parts.append("no credit limit" if limit is None else f"limit ${limit}")
    if data.get("usage") is not None:
        parts.append(f"used ${data['usage']}")
    if data.get("is_free_tier"):
        parts.append("free tier")
    return CheckResult(True, ", ".join(parts) + ".", ms, details)


async def check_openai(ctx: CheckContext) -> CheckResult:
    key = ctx.settings.key_for("openai")
    if not key:
        return CheckResult(False, "OPENAI_API_KEY is not set.", configured=False)
    resp, err, ms = await _get(ctx, "OpenAI", OPENAI_MODELS_URL, {"Authorization": f"Bearer {key}"})
    if resp is None:
        return CheckResult(False, err, ms)
    if resp.status_code != 200:
        return CheckResult(False, status_failure(resp.status_code, "OpenAI", "OPENAI_API_KEY"), ms)
    rows = (_json(resp) or {}).get("data") or []
    count = len(rows) if isinstance(rows, list) else 0
    return CheckResult(True, f"OpenAI accepted the key; {count} models available.", ms, {"models": count})


async def check_typesafe(ctx: CheckContext) -> CheckResult:
    key = ctx.settings.key_for("typesafe")
    if not key:
        return CheckResult(False, "TYPESAFE_API_KEY is not set.", configured=False)
    from typesafe_sdk import (
        AsyncTypeSafeClient,
        Noul,
        RetryPolicy,
        TypeSafeAPIConnectionError,
        TypeSafeAPIError,
        TypeSafeAPITimeoutError,
        TypeSafeError,
    )

    model = ctx.settings.typesafe_model.split(":", 1)[-1] or "jev-latest"
    clock = _Clock()
    try:
        client = AsyncTypeSafeClient(api_key=key, retry=RetryPolicy(max_retries=0), timeout=TIMEOUT_S,
                                     transport=ctx.typesafe_transport)
    except TypeSafeError:
        return CheckResult(False, "TYPESAFE_API_KEY is malformed (whitespace or non-ASCII characters).")
    try:
        async with client:
            response = await client.system_one(
                state="R3 settings page connection check.",
                questions={"check": Noul(instructions="Is this text a connection check?")}, model=model)
    except TypeSafeAPITimeoutError:
        return CheckResult(False, f"TypeSafe did not answer within {int(TIMEOUT_S)} s.", clock.ms)
    except TypeSafeAPIConnectionError as e:
        return CheckResult(False, transport_failure(e, "TypeSafe", "the TypeSafe API"), clock.ms)
    except TypeSafeAPIError as e:
        return CheckResult(False, status_failure(e.status, "TypeSafe", "TYPESAFE_API_KEY"), clock.ms)
    except TypeSafeError as e:
        return CheckResult(False, f"TypeSafe failed: {type(e).__name__}.", clock.ms)
    ms = clock.ms
    return CheckResult(True, f"Jev answered on {response.model} in {ms} ms.", ms, {"model": response.model})


# --- Temporal -------------------------------------------------------------------------


async def check_temporal(ctx: CheckContext) -> CheckResult:
    s = ctx.settings
    if not s.temporal_address:
        return CheckResult(False, "TEMPORAL_ADDRESS is not set: runs execute inside the portal process.",
                           configured=False)
    from temporalio.api.enums.v1 import TaskQueueType
    from temporalio.api.taskqueue.v1 import TaskQueue
    from temporalio.api.workflowservice.v1 import DescribeNamespaceRequest, DescribeTaskQueueRequest
    from temporalio.service import RPCError

    from r3.runs.temporal import connect_client

    connect = ctx.connect or connect_client
    clock = _Clock()
    host = s.temporal_address.rsplit(":", 1)[0]
    try:
        client = await connect(s, timeout=TIMEOUT_S)
    except (TimeoutError, asyncio.TimeoutError):
        return CheckResult(False, f"Temporal at {s.temporal_address} did not answer within {int(TIMEOUT_S)} s.",
                           clock.ms)
    except RPCError as e:
        return CheckResult(False, _temporal_rpc(e, s.temporal_namespace), clock.ms)
    except FileNotFoundError:
        return CheckResult(False, "TEMPORAL_TLS_CERT or TEMPORAL_TLS_KEY names a file that does not exist on the "
                                  "server; paste the PEM text instead.", clock.ms)
    except Exception as e:
        text = str(e)
        if "tls" in text.lower() or "certificate" in text.lower():
            return CheckResult(False, "The TLS handshake with Temporal failed: check the API key or the mTLS "
                                      "certificate and key.", clock.ms)
        if _is_dns(e) or "dns" in text.lower():
            return CheckResult(False, f"Could not resolve {host}: check TEMPORAL_ADDRESS.", clock.ms)
        return CheckResult(False, f"Could not connect to Temporal at {s.temporal_address} ({type(e).__name__}).",
                           clock.ms)
    details: dict[str, Any] = {"address": s.temporal_address, "namespace": s.temporal_namespace,
                               "task_queue": s.temporal_task_queue}
    try:
        async with asyncio.timeout(TIMEOUT_S):
            ns = await client.workflow_service.describe_namespace(
                DescribeNamespaceRequest(namespace=s.temporal_namespace), timeout=timedelta(seconds=TIMEOUT_S))
    except RPCError as e:
        return CheckResult(False, _temporal_rpc(e, s.temporal_namespace), clock.ms, details)
    except Exception as e:
        return CheckResult(False, f"Connected, but describing the namespace failed ({type(e).__name__}).",
                           clock.ms, details)
    info = getattr(ns, "namespace_info", None)
    state = getattr(info, "state", None)
    details["namespace_state"] = _enum_name(state, "NAMESPACE_STATE_")
    pollers = None
    try:
        async with asyncio.timeout(TIMEOUT_S):
            tq = await client.workflow_service.describe_task_queue(
                DescribeTaskQueueRequest(namespace=s.temporal_namespace, task_queue=TaskQueue(name=s.temporal_task_queue),
                                         task_queue_type=TaskQueueType.TASK_QUEUE_TYPE_WORKFLOW),
                timeout=timedelta(seconds=TIMEOUT_S))
        pollers = len(tq.pollers)
    except Exception:
        pass
    details["workers"] = pollers
    ms = clock.ms
    msg = f"Connected to {s.temporal_namespace} ({details['namespace_state'].lower() or 'state unknown'})"
    if pollers is None:
        msg += "; could not read the task queue."
    elif pollers:
        msg += f"; {pollers} worker poller(s) on {s.temporal_task_queue}."
    else:
        msg += f"; no worker is polling {s.temporal_task_queue} yet, so runs would wait."
    return CheckResult(True, msg, ms, details)


def _enum_name(value: Any, prefix: str) -> str:
    try:
        from temporalio.api.enums.v1 import NamespaceState

        name = NamespaceState.Name(value)
    except Exception:
        return ""
    return name.removeprefix(prefix)


def _temporal_rpc(e: Any, namespace: str) -> str:
    from temporalio.service import RPCStatusCode

    status = getattr(e, "status", None)
    if status in (RPCStatusCode.UNAUTHENTICATED, RPCStatusCode.PERMISSION_DENIED):
        return f"Temporal rejected the credentials for namespace {namespace} (check TEMPORAL_API_KEY or the mTLS pair)."
    if status == RPCStatusCode.NOT_FOUND:
        return f"Temporal has no namespace {namespace}: check TEMPORAL_NAMESPACE."
    if status in (RPCStatusCode.UNAVAILABLE, RPCStatusCode.DEADLINE_EXCEEDED):
        return "Temporal is unavailable at that address (check TEMPORAL_ADDRESS and the network)."
    return f"Temporal refused the request ({getattr(status, 'name', 'error')})."


# --- database -------------------------------------------------------------------------------


async def check_database(ctx: CheckContext) -> CheckResult:
    d = ctx.database
    if d is None:
        return CheckResult(False, "No database is open.", configured=False)
    clock = _Clock()
    try:
        async with asyncio.timeout(TIMEOUT_S):
            await d.ping()
            stats = await d.overview()
    except (TimeoutError, asyncio.TimeoutError):
        return CheckResult(False, f"The database did not answer within {int(TIMEOUT_S)} s.", clock.ms)
    except Exception as e:
        return CheckResult(False, f"The database query failed ({type(e).__name__}).", clock.ms)
    ms = clock.ms
    details = {"total_jobs": stats.total_jobs, "new_jobs": stats.new_jobs, "total_sites": stats.total_sites}
    return CheckResult(True, f"Connected: {stats.total_jobs} jobs ({stats.new_jobs} new) from "
                             f"{stats.total_sites} sites.", ms, details)


CHECKS: dict[str, Callable[[CheckContext], Awaitable[CheckResult]]] = {
    "openrouter": check_openrouter,
    "openai": check_openai,
    "typesafe": check_typesafe,
    "temporal": check_temporal,
    "database": check_database,
}


def scrub(value: Any, secrets: list[str]) -> Any:
    """Replace every secret in ``value`` (a string, or nested dicts/lists)."""
    if isinstance(value, str):
        for secret in secrets:
            if secret and secret in value:
                value = value.replace(secret, "••••")
        return value
    if isinstance(value, dict):
        return {k: scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v, secrets) for v in value]
    return value


async def run_check(test_id: str, ctx: CheckContext, secrets: list[str]) -> dict[str, Any]:
    """One check as the API's JSON, scrubbed. Unknown ids raise ``KeyError``."""
    check = CHECKS[test_id]
    try:
        result = await check(ctx)
    except Exception as e:  # a check must never take the page down
        result = CheckResult(False, f"The check failed unexpectedly ({type(e).__name__}).")
    out = result.as_dict(test_id)
    out["message"] = scrub(out["message"], secrets)
    out["details"] = scrub(out["details"], secrets)
    return out
