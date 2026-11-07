"""R3's audit trail: one append-only event per human action and AI decision.

Every event goes two places. The ``r3_events`` table (r3/db/events.py) is what
R3 itself shows: a job's "What happened" and the admin Events page. The
``r3.events`` logger carries the same event as a structured log line, which
r3/obs.py ships to Axiom when it is configured.

A trace id ties the pieces of one decision together: the click in the
browser, the request it sent, the model calls inside that request, the run it
started and the records it changed. The request middleware (r3/web/audit.py)
opens a :func:`trace_scope`; anything recorded inside it, including work in
tasks started from it, carries that trace id and the signed-in account.

What never goes in an event: passwords, tokens, keys, cookies (dropped by key
name at any depth), and anything past the size caps (clipped, so one bulky
value cannot bloat the table).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Literal, Protocol

log = logging.getLogger("r3.events")
_internal = logging.getLogger("r3.events.writer")

Actor = Literal["user", "ai", "system"]

SUMMARY_MAX = 300
STRING_MAX = 2000
DATA_MAX = 8192
MAX_BUFFER = 5000

# Keys whose values are credentials, at any depth: password, new_password,
# api_key, token, session_token, Authorization, cookie, secret...
# Token counts (input_tokens) are kept: only a key ending in "token" is one.
_SECRET_KEY = re.compile(r"pass(word|wd)?|secret|token$|api[_-]?key|authorization|cookie|credential", re.I)


@dataclass
class TraceContext:
    """The request (or run) an event belongs to."""

    trace_id: str
    user_id: int | None = None


_current: ContextVar[TraceContext | None] = ContextVar("r3_trace", default=None)


def new_trace_id() -> str:
    """32 hex characters, the shape of an OpenTelemetry trace id."""
    return secrets.token_hex(16)


def current_trace() -> TraceContext | None:
    return _current.get()


@contextmanager
def trace_scope(trace_id: str | None = None, *, user_id: int | None = None) -> Iterator[TraceContext]:
    """Make ``trace_id`` (a new one when omitted) and the account current for
    everything recorded inside the block."""
    ctx = TraceContext(trace_id or new_trace_id(), user_id)
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


@dataclass
class Event:
    """One row of r3_events."""

    kind: str  # dotted: http.post, page.view, ui.click, job.status, run.started, ai.call, resume.read...
    summary: str  # one plain sentence a person can read
    actor: str = "system"  # user | ai | system
    user_id: int | None = None
    trace_id: str = ""
    subject: str = ""  # what it is about: "job:12", "resume:3", "run:<id>", or ""
    data: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def as_log(self) -> dict[str, Any]:
        out = asdict(self)
        out["at"] = self.at.isoformat()
        return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def scrub(value: Any, depth: int = 0) -> Any:
    """``value`` with credential keys dropped and long strings clipped,
    turned into JSON-safe types."""
    if depth > 6:
        return "…"
    if isinstance(value, dict):
        return {str(k): scrub(v, depth + 1) for k, v in value.items() if not _SECRET_KEY.search(str(k))}
    if isinstance(value, (list, tuple, set)):
        return [scrub(v, depth + 1) for v in list(value)[:100]]
    if isinstance(value, str):
        return _clip(value, STRING_MAX)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    return _clip(str(value), STRING_MAX)


def _bounded(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` when it fits :data:`DATA_MAX` as JSON; otherwise the keys
    whose values fit, in order, marked ``_clipped``."""
    if len(json.dumps(data, default=str)) <= DATA_MAX:
        return data
    out: dict[str, Any] = {"_clipped": True}
    for k, v in data.items():
        trial = {**out, k: v}
        if len(json.dumps(trial, default=str)) <= DATA_MAX:
            out = trial
    return out


class EventStore(Protocol):
    async def insert_events(self, events: list[Event]) -> None: ...


class EventLog:
    """Records events: logs each at once and buffers it for the database,
    which :meth:`flush` (or the :meth:`run` loop) writes in batches. Recording
    never blocks and never raises into the action being recorded."""

    def __init__(self, store: EventStore | None = None, *, max_buffer: int = MAX_BUFFER) -> None:
        self.store = store
        self._buffer: deque[Event] = deque(maxlen=max_buffer)

    def record(self, kind: str, summary: str, *, actor: str = "", subject: str = "",
               data: dict[str, Any] | None = None, user_id: int | None = None,
               trace_id: str | None = None) -> Event:
        ctx = current_trace()
        uid = user_id if user_id is not None else (ctx.user_id if ctx else None)
        ev = Event(kind=kind, summary=_clip(summary.strip(), SUMMARY_MAX), actor=actor or "system",
                   user_id=uid, trace_id=trace_id or (ctx.trace_id if ctx else new_trace_id()),
                   subject=subject, data=_bounded(scrub(data or {})))
        log.info(ev.summary, extra={"event": ev.as_log()})
        if self.store is not None:
            self._buffer.append(ev)
        return ev

    def pending(self) -> list[Event]:
        return list(self._buffer)

    async def flush(self) -> int:
        """Write what is buffered; the number written. A failed write is
        logged and the batch dropped, so a dead database cannot grow memory."""
        if self.store is None or not self._buffer:
            return 0
        batch = list(self._buffer)
        self._buffer.clear()
        try:
            await self.store.insert_events(batch)
        except Exception as e:
            _internal.warning("dropped %d audit events: %s: %s", len(batch), type(e).__name__, e)
            return 0
        return len(batch)


# The process's event log. The portal points it at its database on start
# (r3/web/app.py); code anywhere records through :func:`record`.
default_log = EventLog()


def record(kind: str, summary: str, **kwargs: Any) -> Event:
    return default_log.record(kind, summary, **kwargs)


def job_name(position: str, company: str = "") -> str:
    """How an event names a job: “Position at Company”."""
    return f"{position} at {company}" if company else position
