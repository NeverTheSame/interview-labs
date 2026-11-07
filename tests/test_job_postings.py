"""Who sees a job's full posting (R3_JOB_POSTINGS).

"full" (the default while the owner dogfoods R3): every account sees the
posting, its description and its links, so a job seeker can open the
employer's form, apply there and press "I applied". "teaser": a regular
account gets the OQ-6 teaser instead.
Admins always see the full posting. Runs against the shell stub with real
sign-in; tests/test_job_teaser.py covers the teaser itself."""

from __future__ import annotations

import httpx
import pytest

from r3.settings import ControllerSettings
from r3.web import create_app
from r3.web.auth import SESSION_COOKIE
from tests.fakes import FakeSummarizer
from tests.test_job_teaser import TEASER_JOB, TeaserStub
from tests.test_routes import JOB

# --- the setting -----------------------------------------------------------


def test_full_postings_are_the_default_and_teaser_is_opt_in():
    assert ControllerSettings().job_postings == "full"
    assert ControllerSettings.from_env({}).job_postings == "full"
    assert ControllerSettings.from_env({"R3_JOB_POSTINGS": " Teaser "}).job_postings == "teaser"
    assert ControllerSettings.from_env({"R3_JOB_POSTINGS": "nonsense"}).job_postings == "full"


def test_admins_always_see_postings_and_seekers_only_in_full_mode():
    full, teaser = ControllerSettings(), ControllerSettings(job_postings="teaser")
    assert full.postings_for(is_admin=False) and full.postings_for(is_admin=True)
    assert not teaser.postings_for(is_admin=False) and teaser.postings_for(is_admin=True)


def test_the_setting_can_be_changed_on_the_settings_page():
    from r3.config_registry import BY_NAME

    spec = BY_NAME["R3_JOB_POSTINGS"]
    assert spec.kind == "select" and spec.options == ("full", "teaser") and spec.default == "full"


# --- web --------------------------------------------------------------------


async def _call(method, path, token, monkeypatch, mode):
    if mode is not None:
        monkeypatch.setenv("R3_JOB_POSTINGS", mode)
    else:
        monkeypatch.delenv("R3_JOB_POSTINGS", raising=False)
    app = create_app(database=TeaserStub(), auth=True, job_summarizer=FakeSummarizer())
    transport = httpx.ASGITransport(app=app)
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=transport, base_url="http://test/") as c:
        c.cookies.set(SESSION_COOKIE, token)
        return await c.request(method, path.lstrip("/"))


@pytest.mark.parametrize("mode", [None, "full"])
async def test_in_full_mode_a_seeker_sees_the_posting_and_can_open_it_to_apply(monkeypatch, mode):
    r = await _call("GET", "/jobs/7", "seeker", monkeypatch, mode)
    assert r.status_code == 200
    for shown in ("Open posting", TEASER_JOB.apply_url, JOB.job_url, ">Description<", "Build models",
                  ">I applied</button>", ">Not for me</button>"):
        assert shown in r.text, shown
    # Still a seeker's page: their own buttons, not the operator's status select or trail.
    assert "Set status" not in r.text and "Full trail for this job" not in r.text
    assert "About this role" not in r.text


async def test_in_teaser_mode_a_seeker_gets_the_teaser(monkeypatch):
    r = await _call("GET", "/jobs/7", "seeker", monkeypatch, "teaser")
    assert "About this role" in r.text
    for leak in ("Open posting", JOB.job_url, TEASER_JOB.apply_url, ">Description<"):
        assert leak not in r.text, leak


async def test_api_job_cards_carry_the_link_in_full_mode_only(monkeypatch):
    card = (await _call("GET", "/api/jobs", "seeker", monkeypatch, "full")).json()["jobs"][0]
    assert card["job_url"] == JOB.job_url
    card = (await _call("GET", "/api/jobs", "seeker", monkeypatch, "teaser")).json()["jobs"][0]
    assert card["job_url"] == ""


# --- the assistant ----------------------------------------------------------


async def test_job_details_gives_a_seeker_the_posting_in_full_mode():
    from tests.test_agent_users import DANA, RecordingDB, run_turn, tool_returns

    _, seen = await run_turn(DANA, RecordingDB(), text="tell me about job 7")
    out = tool_returns(seen)["job_details"]
    assert f"URL: {JOB.job_url}" in out and "Description: Build models." in out
    _, seen = await run_turn(DANA, RecordingDB(), text="tell me about job 7", job_postings="teaser")
    out = tool_returns(seen)["job_details"]
    assert "URL:" not in out and "Overview:" in out


async def test_the_brief_matches_the_mode():
    from tests.test_agent_users import DANA, RecordingDB, run_turn

    _, seen = await run_turn(DANA, RecordingDB())
    brief = seen[0][1].instructions
    assert '"Open posting"' in brief and "R3 does not show the original posting" not in brief
    _, seen = await run_turn(DANA, RecordingDB(), job_postings="teaser")
    brief = seen[0][1].instructions
    assert "R3 does not show the original posting" in brief and '"Open posting"' not in brief


async def test_chat_job_cards_carry_the_link_in_full_mode():
    from r3.agent.actions import plan_action
    from r3.agent.contracts import RouteDecision
    from tests.fakes import FakeRuns
    from tests.fakes import test_settings as make_settings
    from tests.test_agent_users import DANA
    from tests.test_routes import StubDB

    async def plan(**settings):
        return await plan_action(RouteDecision(intent="jobs", router="rules", params={}), database=StubDB(),
                                 runs=FakeRuns(), settings=make_settings(**settings),
                                 model_id="openrouter:openrouter/free", text="jobs", user=DANA)

    assert (await plan()).jobs[0].job_url == JOB.job_url
    assert (await plan(job_postings="teaser")).jobs[0].job_url == ""
