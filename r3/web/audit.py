"""The request side of R3's audit trail (r3/events.py).

:class:`AuditTrail` opens a trace for every request and records what the
person did: each POST/PUT/PATCH/DELETE (the route, its job/resume/run, the
outcome and how long it took) and each page opened. Reads the page makes on
its own (the panel's polling, JSON lookups, static files, health checks) are
not actions and are not recorded; the clicks that caused them are, by
static/track.js through /api/events.

The trace id comes from the page when it sends one, so the click, the
request, the model calls inside it and the records it changed share one id.
track.js stamps ``X-R3-Trace`` on the fetch and HTMX requests a click sets
off; a plain form post or link navigates, where no header can be added, so
the click also leaves its id in the short-lived ``r3_trace`` cookie, read
here when the header is absent. The response carries the id back in the
header.

It sits just inside the sign-in gate, so it knows the account; bodies are
never read, so a password or a resume cannot reach an event from here.
"""

from __future__ import annotations

import re
import time
from typing import Any

from r3 import obs
from r3.events import default_log, new_trace_id, trace_scope

from .auth import route_path

TRACE_HEADER = "x-r3-trace"
TRACE_COOKIE = "r3_trace"
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_SKIP_PREFIXES = ("/static/",)
_SKIP_PATHS = frozenset({"/healthz", "/readyz", "/api/events", "/favicon.ico"})
_SUBJECTS = (("job_id", "job"), ("resume_id", "resume"), ("run_id", "run"))


def subject_of(path_params: dict[str, Any]) -> str:
    for param, kind in _SUBJECTS:
        if param in path_params:
            return f"{kind}:{path_params[param]}"
    return ""


def _cookie(header: str, name: str) -> str:
    for part in header.split(";"):
        key, _, value = part.strip().partition("=")
        if key == name:
            return value.strip()
    return ""


def _is_page_view(method: str, path: str, headers: dict[bytes, bytes]) -> bool:
    return (method == "GET" and b"text/html" in headers.get(b"accept", b"")
            and headers.get(b"hx-request") != b"true" and not path.startswith("/api/"))


class AuditTrail:
    """ASGI middleware; plain ASGI so the streamed chat passes through."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = route_path(scope)
        if path in _SKIP_PATHS or path.startswith(_SKIP_PREFIXES):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        sent = headers.get(TRACE_HEADER.encode(), b"").decode("latin-1").strip().lower()
        if not _HEX32.match(sent):
            sent = _cookie(headers.get(b"cookie", b"").decode("latin-1"), TRACE_COOKIE).lower()
        trace_id = sent if _HEX32.match(sent) else new_trace_id()
        user = (scope.get("state") or {}).get("user")
        uid = getattr(user, "id", None)
        method = scope["method"]
        status = {"code": 500}

        async def send_with_trace(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message = {**message, "headers": [*message.get("headers", []),
                                                  (TRACE_HEADER.encode(), trace_id.encode())]}
            await send(message)

        started = time.monotonic()
        with trace_scope(trace_id, user_id=uid), obs.request_span(f"{method} {path}", trace_id):
            try:
                await self.app(scope, receive, send_with_trace)
            finally:
                self._record(scope, path, method, headers, status["code"], started, uid)

    @staticmethod
    def _record(scope: dict[str, Any], path: str, method: str, headers: dict[bytes, bytes], status: int,
                started: float, uid: int | None) -> None:
        ms = int((time.monotonic() - started) * 1000)
        route = getattr(scope.get("route"), "path", "") or path
        actor = "user" if uid is not None else "system"
        if method in ("POST", "PUT", "PATCH", "DELETE"):
            query = scope.get("query_string", b"").decode("latin-1")
            default_log.record(
                f"http.{method.lower()}", f"{method} {route} → {status}", actor=actor,
                subject=subject_of(scope.get("path_params") or {}),
                data={"path": path, "route": route, "query": query, "status": status, "duration_ms": ms,
                      "htmx": headers.get(b"hx-request") == b"true"})
        elif _is_page_view(method, path, headers) and status < 400:
            query = scope.get("query_string", b"").decode("latin-1")
            default_log.record("page.view", f"Opened {path}", actor=actor,
                               subject=subject_of(scope.get("path_params") or {}),
                               data={"path": path, "route": route, "query": query, "status": status,
                                     "duration_ms": ms})
