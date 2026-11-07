"""Who is signed in. Handlers take the account from :func:`current_user`
and never read the session cookie themselves.

:class:`SignInGate` is the only code that reads the cookie. It runs in front
of every route when sign-in is on, puts the account on ``request.state.user``
and turns signed-out requests away: a browser page view is redirected to the
sign-in page, anything else (form posts, JSON, HTMX swaps, streams) gets a
401 so a script never mistakes the login page for its answer.

:func:`require_admin` is the second gate, for the operator console (settings,
labeling): a route or router declares
``dependencies=[Depends(require_admin)]`` and anyone who is not an operator gets
a 403, a short page for a browser page view and JSON for everything else."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote, urlsplit

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

# The model lives with the queries that return it (r3.db cannot import from
# r3.web); this module is still where handlers import it from.
from r3.db.users import SESSION_DAYS, User

__all__ = [
    "LOCAL_USER", "SESSION_COOKIE", "SESSION_MAX_AGE", "OperatorsOnly", "SignInGate", "User", "Viewer",
    "current_user", "operators_only_response", "require_admin", "safe_next", "signup_open", "template_viewer",
]

SESSION_COOKIE = "r3_session"
SESSION_MAX_AGE = SESSION_DAYS * 86400

# The account every request belongs to when sign-in is off (stub-database
# tests, demo servers).
LOCAL_USER = User(id=0, account_name="local", is_admin=True)


def current_user(request: Request) -> User:
    """The signed-in account, set on ``request.state.user`` by the sign-in
    middleware; :data:`LOCAL_USER` when sign-in is off."""
    return getattr(request.state, "user", None) or LOCAL_USER


def signup_open() -> bool:
    """Whether the public sign-up form is on. ``R3_SIGNUP`` of off, 0, false or
    no closes it (accounts are then made with ``python -m r3.accounts``);
    unset keeps it open, as it has always been for a local install. Read at
    call time."""
    return os.environ.get("R3_SIGNUP", "").strip().lower() not in {"off", "0", "false", "no"}


# --- what the page chrome shows -----------------------------------------------


class Viewer(BaseModel):
    """Who the layout is drawn for while sign-in is on: ``user`` is None on
    the signed-out pages (login, sign-up)."""

    user: User | None = None


# Set by the gate for the length of one request so ``layout.html`` can show the
# right header without every handler passing the account to its template.
# Unset (None) means sign-in is off and the layout looks as it always did.
_viewer: ContextVar[Viewer | None] = ContextVar("r3_viewer", default=None)


def template_viewer() -> Viewer | None:
    """The ``viewer()`` template global."""
    return _viewer.get()


# --- the gate -----------------------------------------------------------------

# Reachable signed out: liveness probes, the stylesheet the login page needs,
# and the sign-in flow itself. /logout only clears a cookie, so it is safe too.
_OPEN_PATHS = frozenset({"/healthz", "/readyz", "/login", "/signup", "/logout"})
_OPEN_PREFIXES = ("/static/",)
# Machine endpoints answer 401 even to a GET; a redirect would hand the
# caller an HTML page where it expects JSON or an event stream.
_API_ROOTS = ("/api", "/chat/stream", "/runs", "/stats", "/panel")

_DENIED_BODY = json.dumps({"detail": "Sign in required"}).encode()


def route_path(scope: dict[str, Any]) -> str:
    """The request path without the proxy prefix, as the routes see it."""
    path: str = scope["path"]
    root: str = scope.get("root_path", "")
    if root and (path == root or path.startswith(root + "/")):
        return path[len(root):] or "/"
    return path


def safe_next(value: str | None) -> str:
    """``value`` when it is a path on this site, else ``/``. Guards the
    post-login redirect against being pointed at another host: ``//evil.com``
    and ``/\\evil.com`` are host-relative in browsers, and anything with a
    scheme or control characters is refused outright."""
    if not value or not value.startswith("/") or value.startswith(("//", "/\\")):
        return "/"
    if any(ord(c) < 0x20 or c == "\\" for c in value):
        return "/"
    parts = urlsplit(value)
    if parts.scheme or parts.netloc:
        return "/"
    return value


def login_url(root_path: str, next_path: str | None) -> str:
    """The sign-in page, remembering where to go afterwards."""
    url = f"{root_path}/login"
    if next_path and next_path != "/":
        url += "?next=" + quote(next_path, safe="/")
    return url


def _is_page_view(method: str, path: str, headers: dict[bytes, bytes]) -> bool:
    """Whether a signed-out request is a person opening a page (redirect them)
    rather than a script, a form post or an HTMX swap (refuse it)."""
    if method not in ("GET", "HEAD") or headers.get(b"hx-request") == b"true":
        return False
    return not any(path == r or path.startswith(r + "/") for r in _API_ROOTS)


class SignInGate:
    """ASGI middleware enforcing sign-in. A plain ASGI class rather than
    ``BaseHTTPMiddleware`` so streamed responses (the chat) pass through
    untouched."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = route_path(scope)
        user = None
        if not (path in _OPEN_PATHS or path.startswith(_OPEN_PREFIXES)):
            request = Request(scope)
            token = request.cookies.get(SESSION_COOKIE)
            if token:
                user = await scope["app"].state.db.session_user(token)
            if user is None:
                await self._deny(scope, path, send)
                return
        scope.setdefault("state", {})["user"] = user
        reset = _viewer.set(Viewer(user=user))
        try:
            await self.app(scope, receive, send)
        finally:
            _viewer.reset(reset)

    async def _deny(self, scope: dict[str, Any], path: str, send: Any) -> None:
        root = scope.get("root_path", "")
        headers = dict(scope["headers"])
        if _is_page_view(scope["method"], path, headers):
            query = scope.get("query_string", b"").decode("latin-1")
            target = login_url(root, path + (f"?{query}" if query else ""))
            await _respond(send, 303, b"", [(b"location", target.encode())])
            return
        extra = []
        if headers.get(b"hx-request") == b"true":
            # HTMX follows HX-Redirect even on an error status; send the person
            # back to the page they were on once they have signed in.
            current = urlsplit(headers.get(b"hx-current-url", b"").decode("latin-1"))
            here = route_path({"path": current.path or "/", "root_path": root})
            here += f"?{current.query}" if current.query else ""
            extra.append((b"hx-redirect", login_url(root, safe_next(here)).encode()))
        await _respond(send, 401, _DENIED_BODY, [(b"content-type", b"application/json"), *extra])


# --- operators only -----------------------------------------------------------


class OperatorsOnly(HTTPException):
    """Raised by :func:`require_admin`. An ``HTTPException`` so that, even in an
    app without :func:`operators_only_response` registered, FastAPI still
    answers 403 ``{"detail": "Operators only"}``."""

    def __init__(self) -> None:
        super().__init__(status_code=403, detail="Operators only")


async def require_admin(request: Request) -> User:
    """FastAPI dependency: the signed-in operator, or :class:`OperatorsOnly`.
    With sign-in off everyone is :data:`LOCAL_USER`, an operator, so the console
    behaves as it always did."""
    user = current_user(request)
    if not user.is_admin:
        raise OperatorsOnly()
    return user


async def operators_only_response(request: Request, exc: Exception) -> Response:
    """Exception handler for :class:`OperatorsOnly`: a person opening a page
    gets a short explanation with a way home; scripts, form posts and HTMX
    swaps get JSON they can read."""
    headers = dict(request.scope["headers"])
    if _is_page_view(request.method, route_path(request.scope), headers):
        tmpl = request.app.state.templates.get_template("forbidden.html")
        return HTMLResponse(tmpl.render(title="Operators only", nav=""), status_code=403)
    return JSONResponse({"detail": "Operators only"}, status_code=403)


async def _respond(send: Any, status: int, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [*headers, (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
