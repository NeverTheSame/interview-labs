"""The Controller page as the server renders it (what works with JavaScript
off, and the hooks controller.js relies on), the static files behind it and
a few contract checks on controller.js itself."""

from __future__ import annotations

import re

import pytest
from httpx import ASGITransport, AsyncClient

from r3.runs.models import RunsInfo
from r3.web import create_app
from r3.web.templating import STATIC_DIR
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_routes import StubDB

PREFIX = "/apps/r3"
TABS = ("overview", "runs", "jobs", "stats")


class TemporalRuns(FakeRuns):
    def describe(self) -> RunsInfo:
        return RunsInfo(backend="temporal", configured=True, namespace="r3-prod", task_queue="r3-controller",
                        ui_url="https://cloud.temporal.io/namespaces/r3-prod/workflows")


async def page(settings=None, runs=None, root_path: str = "", database=None) -> str:
    app = create_app(database=database or StubDB(), settings=settings or make_settings(), runs=runs or FakeRuns(),
                     root_path=root_path)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url=f"http://test{root_path}/") as client:
            response = await client.get("/controller")
            assert response.status_code == 200, response.text[:300]
            return response.text


async def test_chat_then_panel_with_four_tabs():
    body = await page()
    assert body.index('id="r3-chat"') < body.index('id="r3-panel"')
    assert 'role="tablist"' in body
    for i, tab in enumerate(TABS):
        assert f'role="tab" id="tab-{tab}" data-tab="{tab}" aria-controls="pane-{tab}"' in body
        assert f'id="pane-{tab}" role="tabpanel" aria-labelledby="tab-{tab}"' in body
        selected = f'aria-controls="pane-{tab}" aria-selected="{"true" if i == 0 else "false"}"'
        assert selected in body
    # Without JavaScript only Overview shows: the pipeline facts and a notice.
    assert re.search(r'id="pane-overview"[^>]*>', body).group(0).count("hidden") == 0
    assert all("hidden" in re.search(rf'id="pane-{t}"[^>]*>', body).group(0) for t in TABS[1:])
    assert "<noscript>" in body and "40</dd>" in body


async def test_base_path_script_and_no_inline_handlers():
    body = await page()
    assert 'data-base-path=""' in body
    assert re.search(r'<script src="/static/controller\.js\?v=[0-9a-f]{10}" defer></script>', body)
    assert "chat.js" not in body
    assert not re.search(r"\son[a-z]+\s*=", body), "no inline event handlers"


async def test_model_picker_renders_the_server_default():
    body = await page()
    select = re.search(r'<select id="chat-model"[^>]*>(.*?)</select>', body, re.S)
    assert select and "disabled" not in select.group(0).split(">")[0]
    options = select.group(1)
    assert '<optgroup label="OpenRouter">' in options and '<optgroup label="OpenAI">' in options
    assert '<option value="openrouter:openrouter/free" data-label="OpenRouter · free router" selected>free router</option>' in options
    assert options.count(" selected") == 1
    assert '<option value="__custom__">Custom OpenRouter model…</option>' in options
    assert '<span id="chat-model-name">OpenRouter · free router</span>' in body
    assert 'data-default-model="openrouter:openrouter/free"' in body


async def test_model_picker_follows_settings():
    body = await page(make_settings(default_model="openai:gpt-4.1", allow_custom_models=False))
    assert '<option value="openai:gpt-4.1" data-label="OpenAI · GPT-4.1" selected>GPT-4.1</option>' in body
    assert "__custom__" not in body


async def test_setup_notice_when_chat_is_not_ready():
    body = await page(make_settings(api_keys={}))
    assert 'data-chat-ready="false"' in body
    assert '<div class="chat-setup" role="status">Set OPENROUTER_API_KEY' in body
    assert re.search(r'<textarea id="chat-input"[^>]*disabled', body)
    assert re.search(r'<select id="chat-model" disabled>\s*<option value="openrouter:openrouter/free" selected>', body)
    assert '<span id="chat-model-name">Setup required</span>' in body


@pytest.mark.parametrize("runs, chip", [
    (FakeRuns(backend="local", reason="No TEMPORAL_ADDRESS"),
     '<a class="chip" id="runner-chip" href="#setup" title="No TEMPORAL_ADDRESS">Local runs</a>'),
    (FakeRuns(configured=False, reason="Runs are switched off"),
     '<a class="chip warn" id="runner-chip" href="#setup" title="Runs are switched off">Runs off</a>'),
    (TemporalRuns(), '<a class="chip ok" id="runner-chip" href="https://cloud.temporal.io/namespaces/r3-prod/workflows" '
                     'target="_blank" rel="noopener" title="Open Temporal">Temporal · r3-prod</a>'),
])
async def test_runner_status_chip(runs, chip):
    body = await page(runs=runs)
    assert chip in body
    assert "TEMPORAL_ADDRESS" in body and "R3_RUNS_BACKEND" in body  # the #setup details
    assert f'data-runs-configured="{"true" if runs.configured else "false"}"' in body


async def test_every_url_stays_under_the_proxy_prefix():
    body = await page(root_path=PREFIX)
    links = re.findall(r'(?:href|src|action|hx-get|hx-post|data-api-url)="(/[^" ]*)"', body)
    assert links and all(link.startswith(PREFIX + "/") for link in links), links
    assert f'data-base-path="{PREFIX}"' in body
    assert f'<script src="{PREFIX}/static/controller.js?v=' in body
    assert f'id="jobs-open" href="{PREFIX}/jobs?status=new"' in body


async def test_starters_exercise_both_systems():
    body = await page()
    for prompt in ("Where do things stand?", "Show me new remote SRE jobs", "Show me new AI/ML jobs from this week",
                   "How should I position myself for platform roles?"):
        assert f'data-prompt="{prompt}"' in body


async def test_controller_js_is_served_and_chat_js_is_gone():
    app = create_app(database=StubDB(), settings=make_settings(), runs=FakeRuns())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            js = await client.get("/static/controller.js")
            assert js.status_code == 200 and "javascript" in js.headers["content-type"]
            assert (await client.get("/static/chat.js")).status_code == 404
    assert not (STATIC_DIR / "chat.js").exists()


def test_controller_js_keeps_its_contract():
    js = (STATIC_DIR / "controller.js").read_text()
    # Server text is never parsed as HTML.
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write", js)
    # Every request goes through the base path: fetch() only ever sees api(...), and API paths only
    # appear as arguments of api() or fetchJSON(), which calls it.
    assert all(arg.startswith("api(") for arg in re.findall(r"\bfetch\(([^,)]*)", js))
    for m in re.finditer(r"[\"`]/api/", js):
        assert re.search(r"(api|fetchJSON)\($", js[max(0, m.start() - 12):m.start()]), js[m.start() - 40:m.start() + 30]
    for event in ("route", "action", "tool_call", "tool_result", "delta", "final", "error"):
        assert f'"{event}"' in js
    for kind in ("jobs", "stats", "unavailable"):
        assert f'case "{kind}"' in js
    for key in ("r3.controller.chat.v1", "r3.controller.model.v1", "r3.controller.tab.v1"):
        assert key in js
    assert 'session.remove("r3.controller.actions.v1")' in js  # the old actions log is cleared, not kept


def test_chat_page_links_accept_same_site_paths_only():
    """OQ-7: a reply's "[Build your resume](/resume/build)" links the page; the
    rule is checked here with Python's regex engine (the browser suite,
    tests/e2e/test_chat_links.py, renders it for real)."""
    js = (STATIC_DIR / "controller.js").read_text()
    m = re.search(r"const pagePath = \(p\) => \(typeof p === \"string\" && /(.+?)/\.test\(p\) \? BASE \+ p : \"\"\);", js)
    assert m, "pagePath must join a vetted path to BASE"
    rule = re.compile(m.group(1))
    for ok in ("/resume/build", "/jobs/1", "/jobs?status=new&role=SRE", "/resume"):
        assert rule.fullmatch(ok), ok
    for bad in ("//evil.example/x", "/\\evil.example", "javascript:alert(1)", "data:text/html,x", "jobs/1",
                "https://evil.example", "/jobs/1 x", "/jobs/\u00011", "/a\\b", ""):
        assert not rule.fullmatch(bad), bad
    assert 'el("a", { class: "md-page", href: page }' in js  # same tab: no target, no outLink
