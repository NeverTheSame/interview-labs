"""Sign-up, sign-in, sign-out and the gate in front of every page.

The accounts live in memory on a StubDB subclass, so nothing here touches a
real database or the network."""

from __future__ import annotations

import re
import secrets

import httpx
import pytest
from fastapi import Request

from r3.db.users import AccountNameTaken, User, password_matches, stored_password
from r3.web import create_app
from r3.web.auth import LOCAL_USER, SESSION_COOKIE, current_user, safe_next
from r3.web.routes_auth import BAD_LOGIN, MISMATCH, MISSING_LOGIN, NAME_RULE, PASSWORD_RULE, TAKEN
from tests.test_routes import StubDB

PREFIX = "/dashboard/controller/r3"


class AuthStubDB(StubDB):
    """StubDB plus the account and session methods of ``UserQueries``."""

    def __init__(self) -> None:
        super().__init__()
        self.accounts: dict[int, tuple[str, str]] = {}  # id -> (account_name, stored password)
        self.sessions: dict[str, int] = {}  # token -> user id
        self.user_tables_ensured = False

    async def ensure_user_tables(self):
        self.user_tables_ensured = True

    async def create_user(self, account_name, password):
        if any(name.lower() == account_name.lower() for name, _ in self.accounts.values()):
            raise AccountNameTaken(account_name)
        uid = len(self.accounts) + 1
        self.accounts[uid] = (account_name, stored_password(password))
        return User(id=uid, account_name=account_name)

    async def check_login(self, account_name, password):
        for uid, (name, stored) in self.accounts.items():
            if name.lower() == account_name.lower() and password_matches(stored, password):
                return User(id=uid, account_name=name)
        return None

    async def create_session(self, user_id):
        token = secrets.token_urlsafe(32)
        self.sessions[token] = user_id
        return token

    async def session_user(self, token):
        uid = self.sessions.get(token)
        return User(id=uid, account_name=self.accounts[uid][0]) if uid else None

    async def delete_session(self, token):
        self.sessions.pop(token, None)


async def _client(app, base_url="http://test"):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url) as c:
            yield c


@pytest.fixture
def stub():
    return AuthStubDB()


@pytest.fixture
async def client(stub):
    async for c in _client(create_app(database=stub, auth=True)):
        yield c


@pytest.fixture
async def prefixed(stub):
    async for c in _client(create_app(database=stub, root_path=PREFIX, auth=True), f"http://test{PREFIX}/"):
        yield c


async def _signup(client, name="Morgan", password="pass1234-long", next_path=None):
    data = {"account_name": name, "password": password, "password_confirm": password}
    if next_path is not None:
        data["next"] = next_path
    return await client.post("signup", data=data)


# --- the gate -----------------------------------------------------------------


@pytest.mark.parametrize("path, location", [
    ("/", "/login"),
    ("/jobs", "/login?next=/jobs"),
    ("/jobs/7?tab=notes", "/login?next=/jobs/7%3Ftab%3Dnotes"),
    ("/settings", "/login?next=/settings"),
])
async def test_signed_out_page_view_redirects_to_login(client, path, location):
    r = await client.get(path)
    assert r.status_code == 303
    assert r.headers["location"] == location


@pytest.mark.parametrize("method, path", [
    ("POST", "/jobs/7/status"),
    ("POST", "/sites/flag"),
    ("GET", "/api/runs"),
    ("GET", "/api/stats"),
    ("GET", "/api/panel"),
    ("POST", "/api/chat/stream"),
    ("GET", "/api/settings"),
])
async def test_signed_out_api_and_posts_get_401_json(client, method, path):
    r = await client.request(method, path)
    assert r.status_code == 401
    assert r.json() == {"detail": "Sign in required"}
    assert "location" not in r.headers and "hx-redirect" not in r.headers


async def test_signed_out_htmx_gets_401_with_hx_redirect_back_to_the_page(client):
    r = await client.get("/jobs?page=2", headers={"HX-Request": "true",
                                                  "HX-Current-URL": "http://test/jobs?status=new"})
    assert r.status_code == 401
    assert r.json() == {"detail": "Sign in required"}
    assert r.headers["hx-redirect"] == "/login?next=/jobs%3Fstatus%3Dnew"


async def test_bogus_cookie_is_signed_out(client):
    client.cookies.set(SESSION_COOKIE, "not-a-session")
    assert (await client.get("/jobs")).status_code == 303


@pytest.mark.parametrize("path", ["/healthz", "/static/app.css", "/login", "/signup"])
async def test_open_paths_need_no_session(client, path):
    assert (await client.get(path)).status_code == 200


async def test_login_page_explains_the_product_and_hides_the_nav(client):
    body = (await client.get("/login")).text
    assert "puts jobs from many job sites in one list" in body
    assert "keeps track of where you applied" in body
    assert 'href="/controller"' not in body and 'href="/jobs"' not in body
    assert 'href="/signup"' in body and "Create account" in body
    assert "Sign out" not in body


# --- sign up ------------------------------------------------------------------


async def test_signup_signs_you_in(client, stub):
    r = await _signup(client)
    assert r.status_code == 303 and r.headers["location"] == "/"
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{SESSION_COOKIE}=")
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie and "Path=/" in cookie
    assert stub.user_tables_ensured
    page = await client.get("/jobs")
    assert page.status_code == 200
    assert 'class="account-name"' in page.text and ">Morgan</span>" in page.text
    assert f'action="/logout"' in page.text and "Sign out" in page.text
    assert 'href="/controller"' in page.text  # the nav is back


@pytest.mark.parametrize("name, password, confirm, error", [
    ("ab", "pass1234-long", "pass1234-long", NAME_RULE),
    ("x" * 41, "pass1234-long", "pass1234-long", NAME_RULE),
    ("bad name", "pass1234-long", "pass1234-long", NAME_RULE),
    ("évelyne", "pass1234-long", "pass1234-long", NAME_RULE),
    ("ok.name_1-x", "zq-pw!", "zq-pw!", PASSWORD_RULE),
    ("ok.name_1-x", "pass1234-long", "pass1234-long5", MISMATCH),
])
async def test_signup_validation_rerenders_the_form(client, stub, name, password, confirm, error):
    r = await client.post("/signup", data={"account_name": name, "password": password,
                                           "password_confirm": confirm, "next": "/jobs"})
    assert r.status_code == 400
    assert error.replace("–", "&ndash;") in r.text or error in r.text
    assert 'role="alert"' in r.text
    assert password not in r.text  # never echoed back
    assert 'name="next" value="/jobs"' in r.text
    assert SESSION_COOKIE not in r.headers.get("set-cookie", "")
    assert stub.accounts == {}


async def test_signup_duplicate_name_is_case_insensitive(client, stub):
    await _signup(client, "Morgan")
    client.cookies.clear()
    r = await _signup(client, "MORGAN", "other-long-pass")
    assert r.status_code == 409
    assert TAKEN in r.text and 'value="MORGAN"' in r.text
    assert len(stub.accounts) == 1


# --- sign in ------------------------------------------------------------------


async def test_login_success_and_case_insensitive_name(client, stub):
    await _signup(client)
    client.cookies.clear()
    r = await client.post("/login", data={"account_name": " morgan ", "password": "pass1234-long", "next": "/jobs/7"})
    assert r.status_code == 303 and r.headers["location"] == "/jobs/7"
    assert (await client.get("/jobs/7")).status_code == 200


@pytest.mark.parametrize("name, password", [("Morgan", "wrong-pass"), ("nobody", "pass1234-long")])
async def test_login_failure_does_not_say_which_part_was_wrong(client, name, password):
    await _signup(client)
    client.cookies.clear()
    r = await client.post("/login", data={"account_name": name, "password": password})
    assert r.status_code == 401
    assert BAD_LOGIN in r.text
    assert "set-cookie" not in r.headers
    assert (await client.get("/jobs")).status_code == 303


async def test_login_missing_fields(client):
    r = await client.post("/login", data={"account_name": "Morgan"})
    assert r.status_code == 400 and MISSING_LOGIN in r.text


@pytest.mark.parametrize("target", [
    "//evil.com", "//evil.com/jobs", "https://evil.com", "http:evil.com", "/\\evil.com", "\\\\evil.com",
    "evil.com", "javascript:alert(1)", "/jobs\r\nSet-Cookie: x=1", "",
])
async def test_next_cannot_leave_the_site(client, target):
    await _signup(client)
    client.cookies.clear()
    r = await client.post("/login", data={"account_name": "Morgan", "password": "pass1234-long", "next": target})
    assert r.status_code == 303 and r.headers["location"] == "/"


async def test_next_keeps_a_local_path_and_query(client):
    r = await _signup(client, next_path="/jobs?status=new&page=2")
    assert r.headers["location"] == "/jobs?status=new&page=2"


def test_safe_next_rules():
    assert safe_next("/jobs/7") == "/jobs/7"
    assert safe_next(None) == "/"
    for bad in ("//evil.com", "/\\evil.com", "https://evil.com", "/\tevil", "relative"):
        assert safe_next(bad) == "/", bad


# --- sign out -----------------------------------------------------------------


async def test_logout_ends_the_session_on_the_server(client, stub):
    await _signup(client)
    token = client.cookies[SESSION_COOKIE]
    assert token in stub.sessions
    r = await client.post("/logout")
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert token not in stub.sessions
    assert f'{SESSION_COOKIE}=""' in r.headers["set-cookie"] and "Max-Age=0" in r.headers["set-cookie"]
    # A copy of the old cookie no longer gets in.
    client.cookies.set(SESSION_COOKIE, token)
    assert (await client.get("/jobs")).status_code == 303


async def test_logout_signed_out_is_harmless(client):
    r = await client.post("/logout")
    assert r.status_code == 303 and r.headers["location"] == "/login"


# --- behind the proxy prefix --------------------------------------------------


async def test_prefix_redirects_cookie_and_links(prefixed):
    r = await prefixed.get("jobs?status=new")
    assert r.status_code == 303
    assert r.headers["location"] == f"{PREFIX}/login?next=/jobs%3Fstatus%3Dnew"
    login = await prefixed.get("login")
    links = re.findall(r'(?:href|src|action)="(/[^" ]*)"', login.text)
    assert links and all(link.startswith(PREFIX + "/") for link in links), links

    r = await _signup(prefixed, next_path="/jobs")
    assert r.headers["location"] == f"{PREFIX}/jobs"
    assert f"Path={PREFIX}" in r.headers["set-cookie"]
    page = await prefixed.get("jobs")
    assert page.status_code == 200
    assert f'action="{PREFIX}/logout"' in page.text
    r = await prefixed.post("logout")
    assert r.headers["location"] == f"{PREFIX}/login"
    assert f"Path={PREFIX}" in r.headers["set-cookie"]


async def test_prefix_open_paths_and_api_401(prefixed):
    assert (await prefixed.get("healthz")).status_code == 200
    assert (await prefixed.get("static/app.css")).status_code == 200
    assert (await prefixed.get("api/runs")).status_code == 401
    r = await prefixed.get("jobs", headers={"HX-Request": "true", "HX-Current-URL": f"http://test{PREFIX}/jobs"})
    assert r.status_code == 401 and r.headers["hx-redirect"] == f"{PREFIX}/login?next=/jobs"


# --- when sign-in is on -------------------------------------------------------


async def test_auth_defaults_off_with_an_injected_database(stub):
    app = create_app(database=stub)
    assert app.state.auth is False
    async for c in _client(app):
        page = await c.get("/jobs")
        assert page.status_code == 200
        assert 'href="/controller"' in page.text
        assert "Sign out" not in page.text and 'href="/login"' not in page.text
        assert (await c.get("/login")).status_code == 404
    assert not stub.user_tables_ensured


def test_auth_defaults_on_when_the_app_opens_its_own_database():
    # Building the app does not connect; the pool opens in the lifespan.
    assert create_app(dsn="postgresql://nobody@127.0.0.1:1/none").state.auth is True
    assert create_app(dsn="postgresql://nobody@127.0.0.1:1/none", auth=False).state.auth is False


async def test_auth_true_forces_the_gate_on_a_stub(client):
    assert (await client.get("/jobs")).status_code == 303


async def test_current_user_is_the_session_account_or_local():
    assert current_user(Request({"type": "http", "headers": []})) == LOCAL_USER
    scope = {"type": "http", "headers": [], "state": {"user": User(id=5, account_name="ann")}}
    assert current_user(Request(scope)) == User(id=5, account_name="ann")
