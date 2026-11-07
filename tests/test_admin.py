"""The operator role: who is an admin, and the console pages only admins reach.

Accounts live in memory (``AdminStubDB``), and the real ``UserQueries`` SQL is
driven through a fake pool, so nothing here touches a database or the network."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from r3.db.users import ADMIN_ACCOUNTS_ENV, UserQueries, account_user, admin_account_names
from r3.portal import auth_enabled
from r3.web import create_app
from r3.web.auth import SESSION_COOKIE
from tests.test_auth import AuthStubDB
from tests.test_routes import StubDB


class AdminStubDB(AuthStubDB):
    """AuthStubDB that keeps ``is_admin`` the way ``r3_users`` does: only an
    account created with ``is_admin=True`` (out of band, never by public
    sign-up) is an operator, and ``R3_ADMIN_ACCOUNTS`` names more."""

    def __init__(self) -> None:
        super().__init__()
        self.admin_ids: set[int] = set()

    def _user(self, user):
        return account_user(user.id, user.account_name, user.id in self.admin_ids) if user else None

    async def create_user(self, account_name, password, *, is_admin=False):
        user = await super().create_user(account_name, password)
        if is_admin:
            self.admin_ids.add(user.id)
        return self._user(user)

    async def check_login(self, account_name, password):
        return self._user(await super().check_login(account_name, password))

    async def session_user(self, token):
        return self._user(await super().session_user(token))


@pytest.fixture(autouse=True)
def _no_env_admins(monkeypatch):
    monkeypatch.delenv(ADMIN_ACCOUNTS_ENV, raising=False)


@pytest.fixture
def stub():
    return AdminStubDB()


async def _client(app):
    # Server errors come back as 500s instead of raising, so a check on the
    # gate is not tripped by an unrelated handler's needs.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


@pytest.fixture
async def client(stub):
    async for c in _client(create_app(database=stub, auth=True)):
        yield c


async def _sign_in_as(client, stub, name: str, *, is_admin: bool = False) -> None:
    """Create the account in the stub and give the client its session."""
    user = await stub.create_user(name, "pass1234", is_admin=is_admin)
    client.cookies.set(SESSION_COOKIE, await stub.create_session(user.id))


@pytest.fixture
async def operator(client, stub):
    await _sign_in_as(client, stub, "Morgan", is_admin=True)
    return client


@pytest.fixture
async def seeker(client, stub):
    await _sign_in_as(client, stub, "seeker")
    return client


# Browser page views: a non-admin sees the "operators only" page.
GATED_PAGES = ["/settings", "/label", "/label/next", "/label/export", "/label/1"]

# Everything else on the console: a non-admin gets 403 JSON.
GATED_CALLS: list[tuple[str, str, dict[str, Any]]] = [
    ("GET", "/api/settings", {}),
    ("PUT", "/api/settings", {"json": {"values": {}}}),
    ("POST", "/api/settings/test/openrouter", {"json": {"values": {}}}),
    ("POST", "/label/1", {"data": {}}),
    ("GET", "/settings", {"headers": {"HX-Request": "true"}}),  # an HTMX swap is not a page view
]

# Job seekers' own pages and endpoints: never refused for want of admin.
UNGATED = ["/", "/controller", "/jobs", "/scout", "/resume", "/api/models", "/api/runs", "/api/stats",
           "/api/panel", "/api/jobs"]


# --- who is an admin ---------------------------------------------------------------------


async def test_no_public_sign_up_is_ever_an_operator(client):
    for name in ("Morgan", "Anna"):  # the first account on an empty table included
        r = await client.post("/signup", data={"account_name": name, "password": "pass1234pass",
                                               "password_confirm": "pass1234pass"})
        assert r.status_code == 303
        assert (await client.get("/settings")).status_code == 403
        await client.post("/logout")
        client.cookies.clear()


async def test_sign_up_cannot_claim_a_name_listed_in_admin_accounts(client, stub, monkeypatch):
    monkeypatch.setenv(ADMIN_ACCOUNTS_ENV, "morgan")
    r = await client.post("/signup", data={"account_name": "Morgan", "password": "pass1234pass",
                                           "password_confirm": "pass1234pass"})
    assert r.status_code == 409 and stub.accounts == {}


async def test_sign_up_can_be_closed(stub, monkeypatch):
    monkeypatch.setenv("R3_SIGNUP", "off")
    async for c in _client(create_app(database=stub, auth=True)):
        assert (await c.get("/signup")).status_code == 403
        r = await c.post("/signup", data={"account_name": "Morgan", "password": "pass1234pass",
                                          "password_confirm": "pass1234pass"})
        assert r.status_code == 403 and stub.accounts == {}
        assert "Create an account" not in (await c.get("/login")).text


async def test_admin_accounts_env_grants_admin_ignoring_case(client, stub, monkeypatch):
    await stub.create_user("Morgan", "pass1234")
    await _sign_in_as(client, stub, "Anna")
    assert (await client.get("/settings")).status_code == 403
    monkeypatch.setenv(ADMIN_ACCOUNTS_ENV, " bob , ANNA")
    assert (await client.get("/settings")).status_code == 200


def test_admin_account_names_parsing(monkeypatch):
    monkeypatch.setenv(ADMIN_ACCOUNTS_ENV, "Morgan, ,ANNA ,")
    assert admin_account_names() == {"morgan", "anna"}
    assert account_user(5, "anna", False).is_admin
    assert account_user(6, "Bob", True).is_admin
    assert not account_user(7, "Bob", False).is_admin
    monkeypatch.delenv(ADMIN_ACCOUNTS_ENV)
    assert admin_account_names() == frozenset()


# --- the real queries, on a fake pool ---------------------------------------------------


class _FakeConn:
    """Answers UserQueries' account SQL from a list of rows, recording what
    ran so the test can check the lock comes first, inside the transaction."""

    def __init__(self, log: list[str], rows: list[dict[str, Any]]) -> None:
        self.log, self.rows = log, rows

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self):
                conn.log.append("BEGIN")

            async def __aexit__(self, *exc):
                conn.log.append("COMMIT" if exc[0] is None else "ROLLBACK")

        return _Tx()

    async def execute(self, sql, *args):
        self.log.append(sql.strip())

    async def fetchrow(self, sql, *args):
        self.log.append(sql.strip())
        if sql.lstrip().startswith("INSERT INTO r3_users"):
            assert "NOT EXISTS" not in sql
            row = {"id": len(self.rows) + 1, "account_name": args[0], "is_admin": args[2]}
            self.rows.append(row)
            return row
        name = args[0].lower()
        return next(({**r, "password": "pass1234"} for r in self.rows if r["account_name"].lower() == name), None)


class _FakePool:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.rows: list[dict[str, Any]] = []
        self.conn = _FakeConn(self.log, self.rows)

    def acquire(self):
        pool = self

        class _Acquire:
            async def __aenter__(self):
                return pool.conn

            async def __aexit__(self, *exc):
                return None

        return _Acquire()

    async def execute(self, sql, *args):
        self.log.append(sql.strip())

    async def fetchrow(self, sql, *args):
        return await self.conn.fetchrow(sql, *args)


def _queries() -> tuple[UserQueries, _FakePool]:
    q = UserQueries()
    q.pool = _FakePool()  # type: ignore[assignment]
    return q, q.pool  # type: ignore[return-value]


async def test_create_user_never_makes_an_operator_unless_asked(monkeypatch):
    q, pool = _queries()
    first = await q.create_user("Morgan", "pass1234pass")
    second = await q.create_user("Anna", "pass1234pass", is_admin=True)
    assert not first.is_admin and second.is_admin
    assert not any("pg_advisory" in s for s in pool.log)
    monkeypatch.setenv(ADMIN_ACCOUNTS_ENV, "morgan")
    assert (await q.check_login("MORGAN", "pass1234")).is_admin
    assert (await q.create_user("bob", "pass1234pass")).is_admin is False


async def test_ensure_user_tables_adds_the_admin_column():
    q, pool = _queries()
    await q.ensure_user_tables()
    assert any("ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false" in s for s in pool.log)


# --- the gate ----------------------------------------------------------------------------


@pytest.mark.parametrize("path", GATED_PAGES)
async def test_non_admin_page_view_gets_the_operators_page(seeker, path):
    response = await seeker.get(path)
    assert response.status_code == 403
    assert response.headers["content-type"].startswith("text/html")
    assert "This page is for R3 operators" in response.text
    assert 'href="/"' in response.text


@pytest.mark.parametrize(("method", "path", "kwargs"), GATED_CALLS)
async def test_non_admin_calls_get_403_json(seeker, method, path, kwargs):
    response = await seeker.request(method, path, **kwargs)
    assert response.status_code == 403
    assert response.json() == {"detail": "Operators only"}


@pytest.mark.parametrize("path", GATED_PAGES)
async def test_admin_reaches_the_console_pages(operator, path):
    assert (await operator.get(path)).status_code not in (401, 403)


@pytest.mark.parametrize(("method", "path", "kwargs"), GATED_CALLS)
async def test_admin_reaches_the_console_calls(operator, method, path, kwargs):
    assert (await operator.request(method, path, **kwargs)).status_code not in (401, 403)


@pytest.mark.parametrize("path", ["/settings"])
async def test_admin_console_pages_render(operator, path):
    assert (await operator.get(path)).status_code == 200


@pytest.mark.parametrize("path", UNGATED)
async def test_job_seeker_pages_stay_open_to_non_admins(seeker, path):
    assert (await seeker.get(path)).status_code not in (401, 403)


async def test_the_operators_page_keeps_the_proxy_prefix(stub):
    prefix = "/apps/r3"
    app = create_app(database=stub, auth=True, root_path=prefix)
    async for c in _client(app):
        await _sign_in_as(c, stub, "seeker")
        response = await c.get(f"{prefix}/settings")
        assert response.status_code == 403
        assert f'href="{prefix}/"' in response.text


@pytest.mark.parametrize("path", GATED_PAGES + UNGATED)
async def test_with_sign_in_off_everything_is_reachable(path):
    async for c in _client(create_app(database=StubDB())):
        assert (await c.get(path)).status_code not in (401, 403)


# --- R3_AUTH ------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "on"), [
    (None, True), ("", True), ("on", True), ("1", True), ("yes", True), ("anything", True),
    ("off", False), ("OFF", False), (" 0 ", False), ("false", False), ("No", False),
])
def test_r3_auth_switch(value, on):
    assert auth_enabled(value) is on
