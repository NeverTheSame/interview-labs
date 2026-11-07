"""Cross-site request refusal and the security headers (``r3.web.security``).

Runs on the stub database, in process: no network, no real database."""

from __future__ import annotations

import httpx
import pytest
from pydantic_ai.models import override_allow_model_requests

from r3.agent.events import DeltaEvent, DoneEvent, parse_sse
from r3.web import create_app
from r3.web.security import is_same_origin
from tests.fakes import FakeRuns, ScriptedAdvisor
from tests.fakes import test_settings as make_settings
from tests.test_auth import AuthStubDB
from tests.test_routes import StubDB

PREFIX = "/dashboard/controller/r3"
EVIL = "http://evil.example"
REFUSED = {"detail": "Cross-site request refused"}

# State-changing endpoints a hostile page would aim a browser at.
UNSAFE_CALLS = [
    ("POST", "/jobs/1/status", {"data": {"status": "applied"}}),
    ("POST", "/jobs/1/notes", {"data": {"notes": "x"}}),
    ("POST", "/sites/flag", {"data": {"url": "https://a.example", "field": "active", "value": "false"}}),
    ("POST", "/api/runs/abc/cancel", {}),
    ("PUT", "/api/settings", {"json": {"values": {}}}),
    ("POST", "/api/chat/stream", {"json": {"messages": [{"role": "user", "content": "hi"}]}}),
]


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


async def _client(app, base_url="http://test"):
    # A refused request must come back as a response; a 500 from a stub that
    # lacks something should not mask what the guard did.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url=base_url) as c:
            yield c


def _app(**kwargs):
    kwargs.setdefault("database", StubDB())
    return create_app(runs=FakeRuns(), advisor=ScriptedAdvisor(make_settings(), [DeltaEvent(text="hello"), DoneEvent()]),
                      **kwargs)


@pytest.fixture
async def client():
    async for c in _client(_app()):
        yield c


# --- the rule --------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "headers", "allowed"), [
    ("POST", {b"host": b"r3.local:8196"}, True),  # no Origin, no Referer: a script
    ("POST", {b"host": b"r3.local:8196", b"origin": b"http://r3.local:8196"}, True),
    ("POST", {b"host": b"r3.local:8196", b"origin": b"http://evil.example"}, False),
    ("POST", {b"host": b"r3.local:8196", b"origin": b"http://r3.local:9999"}, False),  # other port, other origin
    ("POST", {b"host": b"r3.local:8196", b"origin": b"null"}, False),
    ("POST", {b"host": b"r3.local:8196", b"referer": b"http://r3.local:8196/jobs?q=1"}, True),
    ("POST", {b"host": b"r3.local:8196", b"referer": b"http://evil.example/x"}, False),
    # Origin wins over Referer when both are present.
    ("POST", {b"host": b"r3.local", b"origin": b"http://evil.example", b"referer": b"http://r3.local/"}, False),
    ("POST", {b"host": b"r3.example:443", b"origin": b"https://r3.example"}, True),  # default port
    ("POST", {b"host": b"10.0.0.5:8196", b"x-forwarded-host": b"r3.example.com", b"origin": b"https://r3.example.com"},
     True),
    ("POST", {b"host": b"10.0.0.5:8196", b"x-forwarded-host": b"a.example, r3.example.com",
              b"origin": b"https://R3.example.com"}, True),
    ("POST", {b"host": b"10.0.0.5:8196", b"x-forwarded-host": b"r3.example.com", b"origin": b"https://evil.example"},
     False),
    ("PUT", {b"host": b"r3.local", b"origin": b"http://evil.example"}, False),
    ("PATCH", {b"host": b"r3.local", b"origin": b"http://evil.example"}, False),
    ("DELETE", {b"host": b"r3.local", b"origin": b"http://evil.example"}, False),
    ("GET", {b"host": b"r3.local", b"origin": b"http://evil.example"}, True),
    ("HEAD", {b"host": b"r3.local", b"referer": b"http://evil.example/"}, True),
    ("OPTIONS", {b"host": b"r3.local", b"origin": b"http://evil.example"}, True),
])
def test_is_same_origin(method, headers, allowed):
    assert is_same_origin(method, headers) is allowed


# --- on the app ------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path", "kwargs"), UNSAFE_CALLS)
async def test_cross_origin_unsafe_requests_are_refused(client, method, path, kwargs):
    response = await client.request(method, path, headers={"Origin": EVIL}, **kwargs)
    assert response.status_code == 403
    assert response.json() == REFUSED


@pytest.mark.parametrize(("method", "path", "kwargs"), UNSAFE_CALLS)
async def test_cross_site_referer_is_refused_without_origin(client, method, path, kwargs):
    response = await client.request(method, path, headers={"Referer": f"{EVIL}/attack.html"}, **kwargs)
    assert response.status_code == 403


@pytest.mark.parametrize("headers", [{}, {"Origin": "http://test"}, {"Referer": "http://test/sites"}])
async def test_same_origin_and_header_less_posts_pass(client, headers):
    response = await client.post("/sites/flag", headers=headers,
                                 data={"url": "https://a.example", "field": "active", "value": "false"})
    assert response.status_code != 403


async def test_get_is_never_blocked(client):
    response = await client.get("/jobs", headers={"Origin": EVIL, "Referer": f"{EVIL}/x"})
    assert response.status_code == 200


async def test_forwarded_host_is_honoured_behind_a_prefix_proxy():
    async for c in _client(_app(root_path=PREFIX), base_url="http://10.0.0.5:8196"):
        data = {"url": "https://a.example", "field": "active", "value": "false"}
        proxied = {"X-Forwarded-Host": "r3.example.com"}
        ok = await c.post(f"{PREFIX}/sites/flag", data=data, headers={**proxied, "Origin": "https://r3.example.com"})
        assert ok.status_code != 403
        bad = await c.post(f"{PREFIX}/sites/flag", data=data, headers={**proxied, "Origin": EVIL})
        assert bad.status_code == 403 and bad.json() == REFUSED
        settings = await c.get(f"{PREFIX}/settings")
        assert settings.headers["cache-control"] == "no-store"


async def test_refused_before_the_sign_in_gate():
    stub = AuthStubDB()
    async for c in _client(create_app(database=stub, auth=True)):
        refused = await c.post("/jobs/1/status", data={"status": "applied"}, headers={"Origin": EVIL})
        assert refused.status_code == 403 and refused.json() == REFUSED
        # Signed out and same-origin: the gate still answers as before.
        assert (await c.post("/jobs/1/status", data={"status": "applied"})).status_code == 401
        redirect = await c.get("/jobs")
        assert redirect.status_code == 303
        assert redirect.headers["x-frame-options"] == "DENY"


async def test_chat_stream_still_streams_with_the_guard_on(client):
    async with client.stream("POST", "/api/chat/stream", headers={"Origin": "http://test"},
                             json={"messages": [{"role": "user", "content": "hi"}]}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-content-type-options"] == "nosniff"
        body = "".join([chunk async for chunk in response.aiter_text()])
    events = parse_sse(body)
    assert [e.type for e in events] == ["delta", "done"]
    assert events[0].text == "hello"


# --- headers -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/jobs", "/settings", "/api/settings", "/api/stats", "/healthz", "/static/app.css"])
async def test_security_headers_on_every_response(client, path):
    response = await client.get(path)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"


@pytest.mark.parametrize(("path", "no_store"), [
    ("/settings", True), ("/api/settings", True), ("/jobs", False), ("/settingsx", False),
])
async def test_settings_responses_are_not_stored(client, path, no_store):
    response = await client.get(path)
    assert (response.headers.get("cache-control") == "no-store") is no_store


async def test_refusal_carries_the_security_headers(client):
    response = await client.post("/sites/flag", headers={"Origin": EVIL})
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-type"] == "application/json"
