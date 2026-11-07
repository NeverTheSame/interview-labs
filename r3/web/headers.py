"""Response headers that confine what a rendered page can do.

A script that ran in an R3 page would act as the signed-in account. The
Content-Security-Policy below is the second wall behind output escaping and
:func:`r3.text.safe_http_url`: scripts load only from this origin (no inline script, no ``eval``; templates keep their JavaScript in
``static/``), nothing can be embedded as an object, ``<base>`` cannot be
injected, forms post only to this origin, and only this origin may frame a page.

``style-src`` keeps ``'unsafe-inline'`` because templates set per-row bar
widths with ``style="width:..."`` and htmx injects its indicator ``<style>``;
a style can not run code, and the script policy is the strict part.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CSP", "SecurityHeaders"]

CSP = "; ".join((
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'self'",
))

_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"content-security-policy", CSP.encode()),
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"same-origin"),
)


class SecurityHeaders:
    """Plain ASGI middleware (streamed chat responses pass through untouched).
    Adds the headers above to every HTTP response that does not already set
    them."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                out = list(message.get("headers", []))
                present = {k.lower() for k, _ in out}
                out.extend(h for h in _HEADERS if h[0] not in present)
                message = {**message, "headers": out}
            await send(message)

        await self.app(scope, receive, send_with_headers)
