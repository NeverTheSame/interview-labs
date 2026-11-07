"""Sign-up, sign-in and sign-out pages. Mounted only while sign-in is on;
:class:`r3.web.auth.SignInGate` leaves these paths open so a signed-out
visitor can reach them.

A failed form re-renders itself with the message inline and what the visitor
typed (never the password). A failed sign-in never says whether the name or
the password was wrong, so the form cannot be used to discover account names.
"""

from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from r3.db.users import AccountNameTaken, admin_account_names
from r3.events import record as record_event

from .auth import SESSION_COOKIE, SESSION_MAX_AGE, User, safe_next, signup_open

router = APIRouter(tags=["auth"])

NAME_RE = re.compile(r"[A-Za-z0-9._-]{3,40}")
# Applies to new passwords only; an account made with a shorter one keeps signing in.
PASSWORD_MIN = 12

NAME_RULE = "Account name must be 3–40 characters: letters, digits, dot, underscore or dash."
PASSWORD_RULE = f"Password must be at least {PASSWORD_MIN} characters."
MISMATCH = "The two passwords do not match."
TAKEN = "That account name is already taken."
BAD_LOGIN = "Account name or password is incorrect."
MISSING_LOGIN = "Enter your account name and password."
TOO_MANY = "Too many attempts. Try again in a few minutes."
SIGNUP_CLOSED = "Sign-up is closed. Ask the operator for an account."


def _render(request: Request, name: str, status: int = 200, **ctx: Any) -> HTMLResponse:
    tmpl = request.app.state.templates.get_template(f"{name}.html")
    return HTMLResponse(tmpl.render(nav="", query={}, password_min=PASSWORD_MIN, **ctx), status_code=status)


_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def cookie_secure(request: Request) -> bool:
    """Whether the session cookie gets the ``Secure`` flag: always, except for
    a request to localhost or when ``R3_INSECURE_COOKIES=1`` says this
    deployment is plain http on purpose. The request's own scheme is not used:
    behind a TLS-terminating proxy it reads as http."""
    if os.environ.get("R3_INSECURE_COOKIES", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    return (request.url.hostname or "") not in _LOCAL_HOSTS


def _limit_keys(request: Request, scope: str, account_name: str) -> list[str]:
    """What an attempt is counted under: the client address and the account
    name. The address is the socket peer; forwarding headers are not trusted."""
    host = request.client.host if request.client else "unknown"
    return [f"{scope}:ip:{host}", f"{scope}:name:{account_name.lower()}"]


def _throttled(request: Request, name: str, retry_after: int, **ctx: Any) -> HTMLResponse:
    response = _render(request, name, 429, error=TOO_MANY, **ctx)
    response.headers["Retry-After"] = str(retry_after)
    return response


def _signed_in(request: Request, user: User, token: str, next_path: str) -> RedirectResponse:
    """Send the browser on with its new session cookie."""
    root = request.scope.get("root_path", "")
    response = RedirectResponse(f"{root}{safe_next(next_path)}", status_code=303)
    response.set_cookie(
        SESSION_COOKIE, token, max_age=SESSION_MAX_AGE, path=root or "/",
        httponly=True, samesite="lax", secure=cookie_secure(request),
    )
    return response


def _signup_closed(request: Request) -> HTMLResponse:
    return _render(request, "login", 403, title="Sign in", next="/", account_name="", error=SIGNUP_CLOSED,
                   error_field="account_name")


def signup_error(account_name: str, password: str, confirm: str) -> str:
    """The first thing wrong with a sign-up form, or '' when it is valid."""
    if not NAME_RE.fullmatch(account_name):
        return NAME_RULE
    if len(password) < PASSWORD_MIN:
        return PASSWORD_RULE
    if password != confirm:
        return MISMATCH
    return ""


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: str = "/") -> HTMLResponse:
    return _render(request, "login", title="Sign in", next=safe_next(next), account_name="", error="")


@router.post("/login", response_model=None)
async def login(request: Request, account_name: str = Form(""), password: str = Form(""),
                next: str = Form("/")) -> HTMLResponse | RedirectResponse:
    account_name = account_name.strip()
    ctx = {"title": "Sign in", "next": safe_next(next), "account_name": account_name}
    if not account_name or not password:
        return _render(request, "login", 400, error=MISSING_LOGIN,
                       error_field="password" if account_name else "account_name", **ctx)
    db = request.app.state.db
    limiter, keys = request.app.state.auth_limiter, _limit_keys(request, "login", account_name)
    if wait := limiter.retry_after(keys):
        return _throttled(request, "login", wait, error_field="password", **ctx)
    user = await db.check_login(account_name, password)
    if user is None:
        record_event("auth.signin_failed", f"Failed sign-in as “{account_name}”", data={"account_name": account_name})
        limiter.record(keys)  # only failures count, so a good sign-in is never refused
        return _render(request, "login", 401, error=BAD_LOGIN, error_field="password", **ctx)
    record_event("auth.signin", f"{user.account_name} signed in", actor="user", user_id=user.id)
    return _signed_in(request, user, await db.create_session(user.id), next)


@router.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request, next: str = "/") -> HTMLResponse:
    if not signup_open():
        return _signup_closed(request)
    return _render(request, "signup", title="Create account", next=safe_next(next), account_name="", error="")


@router.post("/signup", response_model=None)
async def signup(request: Request, account_name: str = Form(""), password: str = Form(""),
                 password_confirm: str = Form(""), next: str = Form("/")) -> HTMLResponse | RedirectResponse:
    if not signup_open():
        return _signup_closed(request)
    account_name = account_name.strip()
    ctx = {"title": "Create account", "next": safe_next(next), "account_name": account_name}
    limiter, keys = request.app.state.auth_limiter, _limit_keys(request, "signup", account_name)
    if wait := limiter.retry_after(keys):
        return _throttled(request, "signup", wait, error_field="account_name", **ctx)
    limiter.record(keys)
    error = signup_error(account_name, password, password_confirm)
    if error:
        field = {NAME_RULE: "account_name", PASSWORD_RULE: "password"}.get(error, "password_confirm")
        return _render(request, "signup", 400, error=error, error_field=field, **ctx)
    db = request.app.state.db
    if account_name.lower() in admin_account_names():
        # R3_ADMIN_ACCOUNTS grants by name, so a visitor must not be able to
        # register one of those names first; it looks like any taken name.
        return _render(request, "signup", 409, error=TAKEN, error_field="account_name", **ctx)
    try:
        user = await db.create_user(account_name, password)
    except AccountNameTaken:
        return _render(request, "signup", 409, error=TAKEN, error_field="account_name", **ctx)
    record_event("auth.signup", f"{user.account_name} created an account", actor="user", user_id=user.id)
    return _signed_in(request, user, await db.create_session(user.id), next)


@router.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    """End the session on the server, not just in this browser, so a copied
    cookie stops working too."""
    root = request.scope.get("root_path", "")
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        user = await request.app.state.db.session_user(token)
        await request.app.state.db.delete_session(token)
        if user is not None:
            record_event("auth.signout", f"{user.account_name} signed out", actor="user", user_id=user.id)
    response = RedirectResponse(f"{root}/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path=root or "/", httponly=True, samesite="lax",
                           secure=cookie_secure(request))
    return response
