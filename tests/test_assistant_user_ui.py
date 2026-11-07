"""The /controller page for a regular account (the job seeker's "Assistant")
against the operator page admins keep, and the picker's de-duplication."""

from __future__ import annotations

import re

from httpx import ASGITransport, AsyncClient

from r3.db.users import User
from r3.settings import ControllerSettings, ModelChoice
from r3.web import create_app
from r3.web.app import picker_models
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_routes import StubDB

SEEKER = User(id=42, account_name="seeker", is_admin=False)
ADMIN = User(id=7, account_name="boss", is_admin=True)

# Operator internals a regular account must never be shown (finding 1) and
# the operator wording it must not read (finding 2).
OPERATOR_MARKERS = (
    "Runner setup", 'id="setup"', 'id="runner-chip"', "Local runs",
    'id="r3-pipelines"', "Pipeline stages", "Custom OpenRouter model", 'id="chat-model-catalogue"',
    "Snapshot stats", "/presets", "/failures", "TEMPORAL_ADDRESS",
    "<h1>Controller</h1>", "Run the job search from here", "Unreviewed", "unreviewed", "awaiting review",
    ">Pipelines<", "on the right",
)


async def render(user: User | None, settings: ControllerSettings | None = None) -> str:
    app = create_app(database=StubDB(), settings=settings or make_settings(), runs=FakeRuns())
    if user is not None:
        @app.middleware("http")
        async def as_user(request, call_next):  # what the sign-in gate does for a signed-in account
            request.state.user = user
            return await call_next(request)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test/") as client:
            response = await client.get("/controller")
            assert response.status_code == 200, response.text[:300]
            return response.text


def workspace(html: str) -> str:
    """The page body below the site header (layout.html belongs to the shell)."""
    start = html.index('<div class="ctl-head">')
    return html[start:html.index("</aside>", start)]


async def test_regular_account_sees_no_operator_internals():
    body = workspace(await render(SEEKER))
    leaked = [m for m in OPERATOR_MARKERS if m in body]
    assert not leaked, leaked


async def test_regular_account_reads_job_seeker_copy():
    html = await render(SEEKER)
    body = workspace(html)
    assert "<title>Assistant" in html
    assert "<h1>Assistant</h1>" in body
    assert "<h2>Your search</h2>" in body
    tabs = re.findall(r'role="tab"[^>]*>([^<]+)', body)
    assert [t.strip() for t in tabs] == ["Activity", "Jobs", "Your numbers"]
    for prompt in ("Show me new remote DevOps jobs", "Show me AI infrastructure jobs from this week",
                   "How well does my resume fit job #"):
        assert f'data-prompt="{prompt}' in body
    assert 'href="/resume/build"' in body  # "Help me build my resume" is a link, not a prompt
    assert 'href="/resume"' in body
    assert "in the panel" in body


async def test_account_id_is_on_the_page():
    body = workspace(await render(SEEKER))
    assert 'data-user-id="42"' in body and 'data-admin="false"' in body
    admin = workspace(await render(ADMIN))
    assert 'data-user-id="7"' in admin and 'data-admin="true"' in admin


async def test_admin_and_sign_in_off_keep_the_operator_page():
    for user in (ADMIN, None):  # None: sign-in off, the local admin
        body = workspace(await render(user))
        for marker in ("Runner setup", 'id="runner-chip"', 'id="r3-pipelines"',
                       "Custom OpenRouter model", "<h1>Controller</h1>", "<h2>Pipelines</h2>", 'data-tab="overview"'):
            assert marker in body, (user, marker)
    assert 'data-user-id="0"' in workspace(await render(None))


BOTH_KEYS = {"OPENROUTER_API_KEY": "sk-or-test", "OPENAI_API_KEY": "sk-test"}


def test_picker_drops_a_repeated_id():
    settings = ControllerSettings.from_env({**BOTH_KEYS, "R3_CHAT_MODELS": "openai:gpt-4.1-mini,openai:gpt-4.1-mini=Again"})
    ids = [m.id for m in picker_models(settings.available_models(), admin=True)]
    assert ids.count("openai:gpt-4.1-mini") == 1


def test_picker_shows_a_model_once_to_a_regular_account():
    """With both keys the defaults offer GPT-4.1 mini through OpenRouter and
    through OpenAI; the picker strips the provider, so a regular account saw
    "GPT-4.1 mini" twice. Admins keep both routes (in provider groups)."""
    models = ControllerSettings.from_env(BOTH_KEYS).available_models()
    names = [m.label.split(" · ", 1)[-1] for m in models]
    assert names.count("GPT-4.1 mini") == 2  # the cause
    shown = [m.label.split(" · ", 1)[-1] for m in picker_models(models, admin=False)]
    assert shown.count("GPT-4.1 mini") == 1 and len(shown) == len(set(shown))
    assert len(picker_models(models, admin=True)) == len(models)


async def test_regular_picker_is_flat_and_has_no_custom_id():
    body = workspace(await render(SEEKER, make_settings()))
    picker = body[body.index('<select id="chat-model"'):body.index("</select>", body.index('<select id="chat-model"'))]
    assert "<optgroup" not in picker and "__custom__" not in picker
    labels = re.findall(r"<option [^>]*>([^<]+)</option>", picker)
    assert len(labels) == len(set(labels))


def test_picker_with_one_model_hides_the_choice():
    one = [ModelChoice("openrouter:openrouter/free", "OpenRouter · free router")]
    assert picker_models(one, admin=False) == one


def test_controller_js_keys_state_by_account_and_forgets_it_on_sign_out():
    from r3.web.templating import STATIC_DIR
    js = (STATIC_DIR / "controller.js").read_text()
    assert 'ownKey("r3.controller.chat.v1")' in js and 'ownKey("r3.controller.tab.v1")' in js
    assert "root.dataset.userId" in js
    assert 'form[action$="/logout"]' in js


def test_portal_links_refuse_paths_a_browser_reads_as_another_site():
    # "/\evil.example" is "//evil.example" to a browser: another site, not a portal page.
    from r3.web.templating import STATIC_DIR
    js = (STATIC_DIR / "controller.js").read_text()
    assert r'/^\/(?![\/\\])/.test(p)' in js
