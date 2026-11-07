"""Observability: R3's logs and AI traces, shipped to Axiom.

- **Logs.** :class:`AxiomShipper` is a logging handler that batches every log
  record (the audit events of r3/events.py included) and POSTs them to
  Axiom's ingest API as JSON, tagged with ``service`` (r3-portal, r3-worker).
- **AI traces.** :func:`setup_tracing` turns on pydantic-ai's OpenTelemetry
  instrumentation for every agent R3 runs. Spans go to Axiom's OTLP traces
  endpoint; the full prompts and answers ride along only when
  ``AXIOM_AI_CAPTURE_CONTENT=1``, because they hold resumes and other personal
  data. Each model call also becomes an ``ai.call`` audit event (model, tokens,
  time; never the content), so R3's own trail lists every AI decision.

``AXIOM_TOKEN`` and ``AXIOM_DATASET`` turn it on,
``AXIOM_ORG_ID`` and ``AXIOM_URL`` (default https://api.axiom.co) are optional.
Without them nothing is sent anywhere, and audit events still reach Postgres.
Shipping never raises into the app: a failed batch is dropped with a warning
on stderr.
"""

from __future__ import annotations

import json
import logging
import queue
import secrets
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

import httpx
from opentelemetry.sdk.trace import SpanProcessor

from r3 import events as audit

_TRUE = {"1", "true", "yes", "on"}
# Loggers whose lines are the shipper's own traffic; shipping them would loop.
_SELF_LOGGERS = ("httpx", "httpcore", "r3.obs")


@dataclass(frozen=True)
class AxiomConfig:
    token: str
    dataset: str
    url: str = "https://api.axiom.co"
    org_id: str = ""
    capture_content: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "AxiomConfig | None":
        token = (env.get("AXIOM_TOKEN") or "").strip()
        dataset = (env.get("AXIOM_DATASET") or "").strip()
        if not token or not dataset:
            return None
        return cls(token=token, dataset=dataset,
                   url=((env.get("AXIOM_URL") or "").strip() or "https://api.axiom.co").rstrip("/"),
                   org_id=(env.get("AXIOM_ORG_ID") or "").strip(),
                   capture_content=(env.get("AXIOM_AI_CAPTURE_CONTENT") or "").strip().lower() in _TRUE)

    def headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.token}"}
        if self.org_id:
            h["X-Axiom-Org-ID"] = self.org_id
        return h


def otlp_target(cfg: AxiomConfig) -> tuple[str, dict[str, str]]:
    """Axiom's OTLP/HTTP traces endpoint and the headers it needs."""
    headers = {"Authorization": f"Bearer {cfg.token}", "X-Axiom-Dataset": cfg.dataset}
    if cfg.org_id:
        headers["X-Axiom-Org-ID"] = cfg.org_id
    return f"{cfg.url}/v1/traces", headers


# --- logs -----------------------------------------------------------------------

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


def _as_event(record: logging.LogRecord, service: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "_time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        "service": service,
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    for k, v in vars(record).items():
        if k not in _STANDARD_ATTRS and not k.startswith("_"):
            out[k] = v
    if record.exc_info:
        out["exception"] = logging.Formatter().formatException(record.exc_info)
    return out


class AxiomShipper(logging.Handler):
    """Batches log records and POSTs them to Axiom's ingest API from a
    background thread, so logging never waits on the network."""

    def __init__(self, cfg: AxiomConfig, service: str, *, transport: httpx.BaseTransport | None = None,
                 batch: int = 200, interval: float = 2.0, max_queue: int = 20_000, start: bool = True) -> None:
        super().__init__(level=logging.INFO)
        self.cfg = cfg
        self.service = service
        self.batch = batch
        self.interval = interval
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max_queue)
        self._client = httpx.Client(timeout=10.0, transport=transport)
        self._url = f"{cfg.url}/v1/datasets/{cfg.dataset}/ingest"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._warned = 0.0
        self._thread: threading.Thread | None = None
        if start:
            self._thread = threading.Thread(target=self._loop, name="axiom-shipper", daemon=True)
            self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_SELF_LOGGERS):
            return
        try:
            self._queue.put_nowait(_as_event(record, self.service))
        except queue.Full:
            pass  # Axiom is behind; dropping beats growing without bound
        except Exception:
            self.handleError(record)

    def pending(self) -> int:
        return self._queue.qsize()

    def flush(self) -> bool:
        """Send everything queued now; False when a batch could not be sent."""
        ok = True
        with self._lock:
            while True:
                batch: list[dict[str, Any]] = []
                while len(batch) < self.batch:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                if not batch:
                    return ok
                ok = self._send(batch) and ok

    def _send(self, batch: list[dict[str, Any]]) -> bool:
        try:
            resp = self._client.post(self._url, headers={**self.cfg.headers(), "Content-Type": "application/json"},
                                     content=json.dumps(batch, default=str))
            if resp.status_code < 300:
                return True
            reason = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        now = time.monotonic()
        if now - self._warned > 60:
            self._warned = now
            print(f"r3.obs: dropped {len(batch)} log lines for Axiom ({reason})", file=sys.stderr)
        return False

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.flush()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.flush()
        self._client.close()
        super().close()


def setup_logging(service: str, env: Mapping[str, str]) -> AxiomShipper | None:
    """Plain-text logs on stdout, plus Axiom when it is configured."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = AxiomConfig.from_env(env)
    if cfg is None:
        return None
    shipper = AxiomShipper(cfg, service)
    logging.getLogger().addHandler(shipper)
    logging.getLogger("r3.obs").info("logs ship to Axiom dataset %s as service %s", cfg.dataset, service)
    return shipper


# --- AI traces ----------------------------------------------------------------------

_provider: Any = None  # the TracerProvider while tracing is on


class _AiCallEvents(SpanProcessor):
    """Span processor: each model call (a pydantic-ai ``chat`` span) becomes
    an ``ai.call`` audit event in the current request's trace."""

    def __init__(self, events: audit.EventLog | None) -> None:
        self.events = events

    def on_end(self, span: Any) -> None:
        attrs = dict(span.attributes or {})
        if attrs.get("gen_ai.operation.name") != "chat":
            return
        try:
            agent = str(attrs.get("gen_ai.agent.name") or "agent")
            model = str(attrs.get("gen_ai.response.model") or attrs.get("gen_ai.request.model") or "")
            tokens_in = int(attrs.get("gen_ai.usage.input_tokens") or 0)
            tokens_out = int(attrs.get("gen_ai.usage.output_tokens") or 0)
            ms = int(((span.end_time or 0) - (span.start_time or 0)) / 1_000_000)
            failed = getattr(span.status, "status_code", None) is not None and span.status.status_code.name == "ERROR"
            ctx = audit.current_trace()
            log = self.events or audit.default_log
            log.record(
                "ai.call",
                f"{agent} called {model or 'a model'}: {tokens_in} in, {tokens_out} out tokens, "
                f"{ms / 1000:.1f}s" + (" (failed)" if failed else ""),
                actor="ai",
                trace_id=ctx.trace_id if ctx else format(span.context.trace_id, "032x"),
                data={"agent": agent, "provider": attrs.get("gen_ai.provider.name") or attrs.get("gen_ai.system"),
                      "request_model": attrs.get("gen_ai.request.model"), "response_model": model,
                      "input_tokens": tokens_in, "output_tokens": tokens_out, "duration_ms": ms,
                      "failed": failed, "error": span.status.description if failed else None})
        except Exception:
            logging.getLogger("r3.obs").warning("could not record an AI call", exc_info=True)


@dataclass
class TracingHandle:
    provider: Any

    def shutdown(self) -> None:
        global _provider
        from pydantic_ai import Agent

        Agent.instrument_all(False)
        if _provider is self.provider:
            _provider = None
        self.provider.shutdown()


def setup_tracing(service: str, cfg: AxiomConfig | None, *, events: audit.EventLog | None = None) -> TracingHandle:
    """Instrument every pydantic-ai agent. Model calls always become audit
    events; spans go to Axiom only when ``cfg`` is set, with prompts and
    answers only when it says to capture content."""
    global _provider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from pydantic_ai import Agent
    from pydantic_ai.models.instrumented import InstrumentationSettings

    provider = TracerProvider(resource=Resource.create({"service.name": service}))
    provider.add_span_processor(_AiCallEvents(events))
    if cfg is not None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint, headers = otlp_target(cfg)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers)))
    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider,
                                                 include_content=bool(cfg and cfg.capture_content),
                                                 include_binary_content=False))
    _provider = provider
    return TracingHandle(provider)


@contextmanager
def request_span(name: str, trace_id: str) -> Iterator[Any]:
    """A span for one request whose OpenTelemetry trace id is R3's trace id,
    so the model calls inside it land in the same trace in Axiom. Does
    nothing while tracing is off."""
    provider = _provider
    if provider is None:
        yield None
        return
    from opentelemetry import trace
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    try:
        tid = int(trace_id, 16)
    except ValueError:
        tid = int(audit.new_trace_id(), 16)
    parent = trace.set_span_in_context(NonRecordingSpan(SpanContext(
        trace_id=tid, span_id=int(secrets.token_hex(8), 16), is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED))))
    with provider.get_tracer("r3").start_as_current_span(name, context=parent) as span:
        yield span
