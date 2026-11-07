"""The Settings page and API end to end over the app: the page renders, GET
masks secrets, PUT saves, applies and rebuilds what depends on the change,
and every connection test runs against a mocked vendor (httpx.MockTransport,
httpx2.MockTransport for TypeSafe, a fake Temporal client, the stub
database). No secret may appear in any response."""

import json
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from httpx import ASGITransport, AsyncClient

from r3.settings_store import SettingsStore
from r3.web import create_app
from tests import fakes
from tests.test_routes import StubDB

ENV = {
    "R3_SETTINGS_KEY": "test-settings-key-0123456789",
    "OPENAI_API_KEY": "sk-env-openai-key-0000aaaa",
    "R3_ROUTER_MODE": "llm",
}
PORTAL_SECRET = "sk-or-v1-portal-secret-1111bbbb"


class Harness:
    def __init__(self, env=None, db=None):
        self.db = db or StubDB()
        self.store = SettingsStore(env=dict(env or ENV))
        self.runs = fakes.FakeRuns()
        self.advisors: list = []
        self.built_runs: list = []
        self.closed: list = []
        self.app = create_app(database=self.db, settings_store=self.store, runs=self.runs,
                              advisor=fakes.ScriptedAdvisor(self.store.settings()))
        self.app.state.build_advisor = self.build_advisor
        self.app.state.build_runs = self.build_runs

    def build_advisor(self, settings):
        advisor = fakes.ScriptedAdvisor(settings)
        self.advisors.append(advisor)
        return advisor

    async def build_runs(self, settings, database):
        harness = self

        class ClosingRuns(fakes.FakeRuns):
            async def close(self):
                harness.closed.append(self)

        runs = ClosingRuns(backend="temporal" if settings.temporal_configured else "local")
        runs.settings = settings
        self.built_runs.append(runs)
        return runs


@pytest.fixture
async def h():
    harness = Harness()
    async with harness.app.router.lifespan_context(harness.app):
        async with AsyncClient(transport=ASGITransport(app=harness.app), base_url="http://test") as client:
            harness.client = client
            yield harness


def no_secrets(text: str) -> bool:
    return all(s not in text for s in (ENV["OPENAI_API_KEY"], PORTAL_SECRET, "0000aaaa", "portal-secret"))


async def test_page_renders_every_group_and_no_secret(h):
    r = await h.client.get("/settings")
    assert r.status_code == 200 and "<!doctype html>" in r.text
    for title in ("Chat models", "Workflow detection", "Temporal", "Database", "Jobs"):
        assert f">{title}</h2>" in r.text
    assert 'href="/settings" class="on"' in r.text
    assert "/static/settings.css" in r.text and "/static/settings.js" in r.text
    assert 'data-name="OPENAI_API_KEY"' in r.text and "from environment" in r.text  # sources render without JS
    assert "••••aaaa" in r.text and ENV["OPENAI_API_KEY"] not in r.text
    assert "Test all" in r.text and 'data-test="temporal"' in r.text and 'data-test="database"' in r.text


async def test_get_shape_masks_secrets(h):
    r = await h.client.get("/api/settings")
    body = r.json()
    assert no_secrets(r.text)
    assert [g["id"] for g in body["groups"]] == ["models", "router", "temporal", "database", "jobs"]
    keys = {k["name"]: k for g in body["groups"] for k in g["keys"]}
    assert "DATABASE_URL" not in keys
    openai = keys["OPENAI_API_KEY"]
    assert openai["secret"] and openai["set"] and openai["source"] == "env"
    assert openai["value"] is None and openai["masked"] == "••••aaaa"
    assert keys["OPENROUTER_API_KEY"]["set"] is False and keys["OPENROUTER_API_KEY"]["source"] == "default"
    assert keys["R3_ROUTER_MODE"]["value"] == "llm" and keys["R3_ROUTER_MODE"]["options"] == ["rules", "llm", "jev", "hybrid"]
    assert keys["R3_RUNS_BACKEND"]["source"] == "default" and keys["R3_RUNS_BACKEND"]["default"] == "auto"
    assert body["store"] == {"available": True, "error": "", "encryption": "R3_SETTINGS_KEY"}
    tests = {t["id"]: t for g in body["groups"] for t in g["tests"]}
    assert set(tests) == {"openrouter", "openai", "typesafe", "temporal", "database"}
    assert tests["openai"]["configured"] and not tests["openrouter"]["configured"]


async def test_put_saves_applies_and_rebuilds_the_advisor(h):
    r = await h.client.put("/api/settings", json={"values": {"OPENROUTER_API_KEY": PORTAL_SECRET,
                                                             "R3_ROUTER_MODE": "rules"}})
    assert r.status_code == 200, r.text
    assert no_secrets(r.text)
    body = r.json()
    apply = body["apply"]
    assert apply["ok"] and sorted(apply["changed"]) == ["OPENROUTER_API_KEY", "R3_ROUTER_MODE"]
    assert apply["rebuilt"] == ["settings", "advisor"]  # no Temporal change: runs untouched
    keys = {k["name"]: k for g in body["groups"] for k in g["keys"]}
    assert keys["OPENROUTER_API_KEY"]["source"] == "portal" and keys["OPENROUTER_API_KEY"]["masked"] == "••••bbbb"
    # Live immediately: the settings object, the advisor, and what routes read.
    live = h.app.state.settings
    assert live.router_mode == "rules" and live.key_for("openrouter") == PORTAL_SECRET
    assert h.app.state.advisor is h.advisors[-1] and h.advisors[-1].settings is live
    assert h.app.state.runs is h.runs and not h.built_runs
    models = (await h.client.get("/api/models")).json()
    assert models["providers"]["openrouter"] is True
    # Persisted, encrypted, and the secret never logged into the stub's writes.
    stored, secret = h.db.settings_rows["OPENROUTER_API_KEY"]
    assert secret and PORTAL_SECRET not in stored
    assert ("setting", "OPENROUTER_API_KEY") in h.db.writes


async def test_put_temporal_keys_rebuilds_runs_and_closes_the_old_ones(h):
    r = await h.client.put("/api/settings", json={"values": {"TEMPORAL_ADDRESS": "localhost:7233"}})
    apply = r.json()["apply"]
    assert apply["rebuilt"][:3] == ["settings", "advisor", "runs"]
    assert h.app.state.runs is h.built_runs[0] and h.built_runs[0].settings.temporal_address == "localhost:7233"
    assert apply["runs"]["backend"] == "temporal" and "Runs now execute Temporal (test)." in apply["messages"]
    assert h.app.state.owns_runs is True and h.closed == []  # the injected runner is not ours to close

    await h.client.put("/api/settings", json={"values": {"TEMPORAL_TASK_QUEUE": "other-queue"}})
    assert h.app.state.runs is h.built_runs[1] and h.closed == [h.built_runs[0]]  # the one we built is closed
    assert h.built_runs[1].settings.temporal_task_queue == "other-queue"


async def test_clearing_an_override_returns_to_env(h):
    await h.client.put("/api/settings", json={"values": {"OPENAI_API_KEY": "sk-portal-openai-2222cccc"}})
    assert h.app.state.settings.key_for("openai") == "sk-portal-openai-2222cccc"
    r = await h.client.put("/api/settings", json={"values": {"OPENAI_API_KEY": None}})
    keys = {k["name"]: k for g in r.json()["groups"] for k in g["keys"]}
    assert keys["OPENAI_API_KEY"]["source"] == "env" and keys["OPENAI_API_KEY"]["masked"] == "••••aaaa"
    assert h.app.state.settings.key_for("openai") == ENV["OPENAI_API_KEY"]
    assert "OPENAI_API_KEY" not in h.db.settings_rows


async def test_put_rejects_invalid_and_unknown_values_atomically(h):
    r = await h.client.put("/api/settings", json={"values": {"R3_ROUTER_MODE": "rules", "DATABASE_URL": "x",
                                                             "R3_JEV_CONFIDENCE": "2"}})
    assert r.status_code == 400
    errors = r.json()["detail"]["errors"]
    assert set(errors) == {"DATABASE_URL", "R3_JEV_CONFIDENCE"}
    assert not getattr(h.db, "settings_rows", {}) and h.app.state.settings.router_mode == "llm"


async def test_overrides_load_at_startup():
    db = StubDB()
    first = Harness(db=db)
    async with first.app.router.lifespan_context(first.app):
        async with AsyncClient(transport=ASGITransport(app=first.app), base_url="http://test") as c:
            await c.put("/api/settings", json={"values": {"R3_ROUTER_MODE": "hybrid", "OPENROUTER_API_KEY": PORTAL_SECRET}})
    second = Harness(db=db)  # a restart: same database, same key
    async with second.app.router.lifespan_context(second.app):
        assert second.app.state.settings.router_mode == "hybrid"
        assert second.app.state.settings.key_for("openrouter") == PORTAL_SECRET


async def test_pinned_settings_refuse_to_save():
    settings = fakes.test_settings()
    app = create_app(database=StubDB(), settings=settings, runs=fakes.FakeRuns(),
                     advisor=fakes.ScriptedAdvisor(settings))
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            body = (await c.get("/api/settings")).json()
            assert body["pinned"] and not body["store"]["available"]
            r = await c.put("/api/settings", json={"values": {"R3_ROUTER_MODE": "rules"}})
            assert r.status_code == 503 and "overrides are off" in r.json()["detail"]
            assert "overrides are off" in (await c.get("/settings")).text


# --- connection tests -------------------------------------------------------------------


def mock(h, handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    h.app.state.check_transport = httpx.MockTransport(record)
    return seen


async def post_test(h, test_id, values=None):
    r = await h.client.post(f"/api/settings/test/{test_id}", json={"values": values or {}})
    assert r.status_code == 200, r.text
    assert no_secrets(r.text) and "typed-key-9999dddd" not in r.text
    body = r.json()
    assert set(body) >= {"ok", "message", "latency_ms", "details"} and isinstance(body["latency_ms"], int)
    return body


async def test_openrouter_uses_the_typed_candidate_and_reports_the_key(h):
    seen = mock(h, lambda r: httpx.Response(200, json={"data": {"label": "r3-prod", "limit": 25, "usage": 1.5,
                                                                "is_free_tier": False}}))
    body = await post_test(h, "openrouter", {"OPENROUTER_API_KEY": "typed-key-9999dddd"})
    assert body["ok"] and "label r3-prod" in body["message"] and "limit $25" in body["message"]
    assert body["details"]["limit"] == 25
    assert str(seen[0].url) == "https://openrouter.ai/api/v1/auth/key"
    assert seen[0].headers["authorization"] == "Bearer typed-key-9999dddd"
    assert h.app.state.settings.key_for("openrouter") == ""  # testing saves nothing


async def test_openrouter_unset_and_rejected(h):
    body = await post_test(h, "openrouter")
    assert not body["ok"] and body["configured"] is False and "not set" in body["message"]
    mock(h, lambda r: httpx.Response(401, json={"error": {"message": "No auth credentials found"}}))
    body = await post_test(h, "openrouter", {"OPENROUTER_API_KEY": "typed-key-9999dddd"})
    assert not body["ok"] and body["message"] == "OpenRouter rejected OPENROUTER_API_KEY (HTTP 401)."


async def test_a_vendor_echoing_the_key_is_scrubbed(h):
    mock(h, lambda r: httpx.Response(200, json={"data": {"label": "copy typed-key-9999dddd", "limit": None}}))
    body = await post_test(h, "openrouter", {"OPENROUTER_API_KEY": "typed-key-9999dddd"})
    assert body["ok"] and "••••" in body["details"]["label"]


async def test_openai_counts_models_with_the_stored_key(h):
    seen = mock(h, lambda r: httpx.Response(200, json={"data": [{"id": "gpt-4.1"}, {"id": "gpt-4.1-mini"}]}))
    body = await post_test(h, "openai")
    assert body["ok"] and "2 models" in body["message"] and body["details"] == {"models": 2}
    assert seen[0].headers["authorization"] == f"Bearer {ENV['OPENAI_API_KEY']}"


async def test_openai_timeout_and_dns_failures_read_well(h):
    def slow(request):
        raise httpx.ReadTimeout("timed out", request=request)

    mock(h, slow)
    body = await post_test(h, "openai")
    assert not body["ok"] and body["message"] == "OpenAI did not answer within 10 s."

    def dns(request):
        raise httpx.ConnectError("[Errno -2] Name or service not known", request=request)

    mock(h, dns)
    body = await post_test(h, "openai")
    assert "Could not resolve api.openai.com" in body["message"]


async def test_typesafe_asks_one_question(h):
    sent = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json={"model": "jev-1.13.0", "answers": {"check": {"type": "noul", "noul": 0.97}},
                                          "usage": {"input_tokens": 12, "output_tokens": 1}})

    h.app.state.check_typesafe_transport = httpx2.MockTransport(handler)
    body = await post_test(h, "typesafe", {"TYPESAFE_API_KEY": "typed-key-9999dddd"})
    assert body["ok"] and body["message"].startswith("Jev answered on jev-1.13.0")
    assert body["details"] == {"model": "jev-1.13.0"}
    assert len(sent) == 1 and sent[0]["questions"]["check"]["type"] == "noul"

    h.app.state.check_typesafe_transport = httpx2.MockTransport(lambda r: httpx2.Response(401, json={"error": "bad key"}))
    body = await post_test(h, "typesafe", {"TYPESAFE_API_KEY": "typed-key-9999dddd"})
    assert not body["ok"] and body["message"] == "TypeSafe rejected TYPESAFE_API_KEY (HTTP 401)."


class FakeWorkflowService:
    def __init__(self, pollers):
        self.pollers = pollers
        self.requests = []

    async def describe_namespace(self, request, timeout=None):
        self.requests.append(request)
        return SimpleNamespace(namespace_info=SimpleNamespace(state=1))

    async def describe_task_queue(self, request, timeout=None):
        self.requests.append(request)
        return SimpleNamespace(pollers=[object()] * self.pollers)


async def test_temporal_connects_describes_and_counts_workers(h):
    service = FakeWorkflowService(pollers=2)
    connected = []

    async def connect(settings, *, timeout):
        connected.append(settings)
        return SimpleNamespace(workflow_service=service)

    h.app.state.check_temporal_connect = connect
    body = await post_test(h, "temporal", {"TEMPORAL_ADDRESS": "localhost:7233", "TEMPORAL_NAMESPACE": "r3-ns"})
    assert body["ok"] and body["message"] == "Connected to r3-ns (registered); 2 worker poller(s) on r3-controller."
    assert body["details"]["workers"] == 2 and connected[0].temporal_namespace == "r3-ns"
    assert service.requests[0].namespace == "r3-ns"

    service.pollers = 0
    body = await post_test(h, "temporal", {"TEMPORAL_ADDRESS": "localhost:7233"})
    assert body["ok"] and "no worker is polling r3-controller" in body["message"]

    async def refuse(settings, *, timeout):
        raise TimeoutError

    h.app.state.check_temporal_connect = refuse
    body = await post_test(h, "temporal", {"TEMPORAL_ADDRESS": "localhost:7233"})
    assert not body["ok"] and "did not answer within 10 s" in body["message"]


async def test_temporal_unset(h):
    body = await post_test(h, "temporal")
    assert not body["ok"] and body["configured"] is False


async def test_database_check(h):
    body = await post_test(h, "database")
    assert body["ok"] and body["message"] == "Connected: 100 jobs (40 new) from 4 sites."
    h.db.ping_ok = False
    body = await post_test(h, "database")
    assert not body["ok"] and "ConnectionError" in body["message"]


async def test_unknown_test_and_invalid_candidate(h):
    assert (await h.client.post("/api/settings/test/smtp", json={"values": {}})).status_code == 404
    r = await h.client.post("/api/settings/test/temporal", json={"values": {"TEMPORAL_ADDRESS": "no port"}})
    assert r.status_code == 400 and "TEMPORAL_ADDRESS" in r.json()["detail"]["errors"]


async def test_a_settings_save_is_audited_by_name_never_by_value(h):
    from r3.events import default_log
    from tests.test_events import MemoryStore

    store, previous = MemoryStore(), default_log.store
    default_log.store = store
    try:
        await h.client.put("/api/settings", json={"values": {"OPENROUTER_API_KEY": PORTAL_SECRET,
                                                             "R3_ROUTER_MODE": "rules"}})
        await default_log.flush()
    finally:
        default_log.store = previous
    ev = next(e for e in store.rows if e.kind == "settings.changed")
    assert ev.actor == "user" and ev.data == {"changed": ["OPENROUTER_API_KEY", "R3_ROUTER_MODE"]}
    assert ev.summary == "Settings changed: OPENROUTER_API_KEY, R3_ROUTER_MODE"
    assert PORTAL_SECRET not in repr(store.rows)
