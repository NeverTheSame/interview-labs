"""Browser-facing protections that apply to every request, signed in or not.

:class:`SameOriginGuard` refuses state-changing requests that a browser sent
on behalf of another site. A form or ``fetch`` on ``evil.example`` can make the
browser POST here with the person's session cookie; the browser always says
where such a request came from (``Origin``, or at least ``Referer``), and the
page cannot forge either. So an unsafe request whose stated origin is not this
host is refused, and one that states no origin at all (curl, scripts, tests)
passes: it carries no ambient browser credentials to abuse.

The same middleware adds the response headers every page should carry, and
``Cache-Control: no-store`` on the settings page and API, which show API keys.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlsplit

from .auth import route_path

__all__ = ["SameOriginGuard", "is_same_origin"]

_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_SECURITY_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"same-origin"),
)

# Pages and endpoints whose responses hold secrets: never cached anywhere.
_NO_STORE_ROOTS = ("/settings", "/api/settings")

_REFUSED_BODY = json.dumps({"detail": "Cross-site request refused"}).encode()


def _host(value: str) -> str:
    """``host[:port]`` lower-cased, without the port when it is a default one,
    so ``Origin: https://r3.example`` matches ``Host: r3.example:443``."""
    host = value.strip().lower()
    for default in (":80", ":443"):
        if host.endswith(default):
            return host[: -len(default)]
    return host


def _stated_origin(headers: dict[bytes, bytes]) -> str | None:
    """The host the browser says the request came from, ``""`` when it states
    one that cannot be matched (``Origin: null`` from a sandboxed frame or a
    file), or None when it states nothing."""
    origin = headers.get(b"origin")
    if origin is not None:
        value = origin.decode("latin-1").strip()
        return "" if value in ("", "null") else _host(urlsplit(value).netloc)
    referer = headers.get(b"referer")
    if referer is not None:
        return _host(urlsplit(referer.decode("latin-1").strip()).netloc)
    return None


def _own_hosts(headers: dict[bytes, bytes]) -> set[str]:
    """The names this server is being reached by: ``Host`` and, behind a
    proxy, every entry of ``X-Forwarded-Host``. A browser cannot set the
    latter on a cross-site request without a CORS preflight R3 never grants."""
    hosts = {_host(headers.get(b"host", b"").decode("latin-1"))}
    forwarded = headers.get(b"x-forwarded-host", b"").decode("latin-1")
    hosts.update(_host(h) for h in forwarded.split(",") if h.strip())
    hosts.discard("")
    return hosts


def is_same_origin(method: str, headers: dict[bytes, bytes]) -> bool:
    """Whether a request may proceed: safe methods always, unsafe ones when
    they state no origin or one of this server's own hosts."""
    if method.upper() not in _UNSAFE_METHODS:
        return True
    stated = _stated_origin(headers)
    return stated is None or stated in _own_hosts(headers)


def _wants_no_store(path: str) -> bool:
    return any(path == r or path.startswith(r + "/") for r in _NO_STORE_ROOTS)


class SameOriginGuard:
    """ASGI middleware: cross-site refusal plus security headers. A plain
    ASGI class rather than ``BaseHTTPMiddleware`` so streamed responses (the
    chat) pass through untouched; it only edits the response's start
    message."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        if not is_same_origin(scope["method"], headers):
            await send({"type": "http.response.start", "status": 403, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(_REFUSED_BODY)).encode()),
                *_SECURITY_HEADERS,
            ]})
            await send({"type": "http.response.body", "body": _REFUSED_BODY})
            return
        no_store = _wants_no_store(route_path(scope))

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                out = list(message.get("headers", []))
                if no_store:
                    out = [(k, v) for k, v in out if k.lower() != b"cache-control"]
                    out.append((b"cache-control", b"no-store"))
                present = {k.lower() for k, _ in out}
                out.extend(h for h in _SECURITY_HEADERS if h[0] not in present)
                message = {**message, "headers": out}
            await send(message)

        await self.app(scope, receive, send_with_headers)
