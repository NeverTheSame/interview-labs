"""Sign-in and sign-up throttling, the password minimum, and the session cookie flags."""

from __future__ import annotations

import httpx
import pytest

from r3.web import create_app
from r3.web.ratelimit import RateLimiter
from r3.web.routes_auth import PASSWORD_MIN, PASSWORD_RULE
from tests.test_auth import AuthStubDB

GOOD = "a-good-long-password"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# --- the limiter --------------------------------------------------------------------------


def test_limiter_blocks_after_the_limit_and_frees_after_the_window():
    clock = Clock()
    lim = RateLimiter(limit=3, window=60, clock=clock)
    for _ in range(3):
        assert lim.retry_after(["k"]) == 0
        lim.record(["k"])
    assert lim.retry_after(["k"]) == 60
    clock.now += 59
    assert lim.retry_after(["k"]) == 1
    clock.now += 2
    assert lim.retry_after(["k"]) == 0


def test_limiter_keys_are_independent_and_any_blocked_key_blocks():
    lim = RateLimiter(limit=1, window=60, clock=Clock())
    lim.record(["ip:1", "name:a"])
    assert lim.retry_after(["ip:2", "name:a"]) > 0  # same name, other address
    assert lim.retry_after(["ip:1", "name:b"]) > 0  # same address, other name
    assert lim.retry_after(["ip:2", "name:b"]) == 0


def test_limiter_forgets_idle_keys():
    clock = Clock()
    lim = RateLimiter(limit=1, window=60, clock=clock)
    lim.record(["k"])
    clock.now += 61
    lim.retry_after(["other"])
    assert "k" not in lim._hits


# --- the routes ---------------------------------------------------------------------------


@pytest.fixture
def stub():
    return AuthStubDB()


async def _client(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest.fixture
async def clocked(stub):
    app = create_app(database=stub, auth=True)
    app.state.auth_limiter = RateLimiter(limit=3, window=900, clock=(clock := Clock()))
    async for c in _client(app):
        c.clock = clock  # type: ignore[attr-defined]
        yield c


async def test_failed_logins_are_throttled_per_account_name(stub, clocked):
    await stub.create_user("Morgan", GOOD)
    for _ in range(3):
        assert (await clocked.post("/login", data={"account_name": "morgan", "password": "wrong"})).status_code == 401
    r = await clocked.post("/login", data={"account_name": "MORGAN", "password": GOOD})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert "Too many" in r.text
    clocked.clock.now += 901
    assert (await clocked.post("/login", data={"account_name": "morgan", "password": GOOD})).status_code == 303


async def test_failed_logins_are_throttled_per_client_address_across_names(clocked):
    for i in range(3):
        await clocked.post("/login", data={"account_name": f"user{i}", "password": "wrong"})
    r = await clocked.post("/login", data={"account_name": "someone-new", "password": "wrong"})
    assert r.status_code == 429


async def test_good_logins_do_not_count(stub, clocked):
    await stub.create_user("Morgan", GOOD)
    for _ in range(6):
        assert (await clocked.post("/login", data={"account_name": "morgan", "password": GOOD})).status_code == 303


async def test_sign_ups_are_throttled_per_address(clocked):
    for i in range(3):
        r = await clocked.post("/signup", data={"account_name": f"user{i}", "password": GOOD, "password_confirm": GOOD})
        assert r.status_code == 303
    r = await clocked.post("/signup", data={"account_name": "user9", "password": GOOD, "password_confirm": GOOD})
    assert r.status_code == 429


async def test_new_passwords_need_twelve_characters(clocked, stub):
    assert PASSWORD_MIN == 12
    r = await clocked.post("/signup", data={"account_name": "morgan", "password": "x" * 11, "password_confirm": "x" * 11})
    assert r.status_code == 400 and PASSWORD_RULE in r.text and stub.accounts == {}
    # The form's hint states the same rule before anyone types.
    form = await clocked.get("/signup")
    assert f"At least {PASSWORD_MIN} characters." in form.text and "At least 4 characters" not in form.text


async def test_an_existing_short_password_still_signs_in(stub, clocked):
    stub.accounts[1] = ("old", "abcd")  # a legacy plain-text account with a 4-character password
    r = await clocked.post("/login", data={"account_name": "old", "password": "abcd"})
    assert r.status_code == 303


# --- the cookie ---------------------------------------------------------------------------


async def _cookie_for(stub, base_url, monkeypatch, headers=None, insecure=None):
    if insecure is None:
        monkeypatch.delenv("R3_INSECURE_COOKIES", raising=False)
    else:
        monkeypatch.setenv("R3_INSECURE_COOKIES", insecure)
    app = create_app(database=stub, auth=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url) as c:
            r = await c.post("/signup", data={"account_name": "user" + str(len(stub.accounts)), "password": GOOD,
                                              "password_confirm": GOOD}, headers=headers)
    return r.headers["set-cookie"]


@pytest.mark.parametrize("base, secure", [
    ("http://app.example", True),                    # plain http behind a proxy: still Secure
    ("https://app.example", True),
    ("http://localhost:8090", False),
    ("http://127.0.0.1:8090", False),
    ("http://[::1]:8090", False),
])
async def test_cookie_is_secure_except_on_localhost(stub, monkeypatch, base, secure):
    cookie = await _cookie_for(stub, base, monkeypatch)
    assert ("Secure" in cookie) is secure
    assert "HttpOnly" in cookie and "SameSite=lax" in cookie


async def test_insecure_cookies_can_be_allowed_explicitly(stub, monkeypatch):
    assert "Secure" not in await _cookie_for(stub, "http://app.example", monkeypatch, insecure="1")
    assert "Secure" in await _cookie_for(stub, "http://app.example", monkeypatch, insecure="0")
