"""The assistant is per account: tools read the signed-in user's rows,
regular users are held to the model list, runs started from chat carry the
account, and the prompt describes only the product that exists. Deterministic: FunctionModel, a recording stub database, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from r3.agent.actions import plan_action
from r3.agent.contracts import ChatTurnRequest, RouteDecision, StatsAction
from r3.agent.service import Advisor, turn_model
from r3.db.resumes import Resume, ResumeSummary
from r3.db.users import User
from r3.web import routes_chat
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_chat_stream import client_for
from tests.test_routes import JOB, StubDB

DANA = User(id=7, account_name="dana", is_admin=False)
ERIK = User(id=8, account_name="erik", is_admin=False)
ADMIN = User(id=1, account_name="root", is_admin=True)
NOW = datetime.now(timezone.utc)
RESUME_TEXT = "Dana Doe - Platform engineer. Kubernetes, Terraform, AWS. " * 200  # ~11k chars: must be clipped


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


class RecordingDB(StubDB):
    """StubDB that records the ``user_id`` of every per-user query and has a
    resume for DANA only."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, Any]] = []

    async def overview(self, *, user_id=None):
        self.calls.append(("overview", user_id))
        return await super().overview(user_id=user_id)

    async def job_facets(self, *, user_id=None):
        self.calls.append(("job_facets", user_id))
        return await super().job_facets(user_id=user_id)

    async def list_jobs(self, f, page, size, *, user_id=None):
        self.calls.append(("list_jobs", user_id))
        return await super().list_jobs(f, page, size, user_id=user_id)

    async def get_job(self, job_id, *, user_id=None):
        self.calls.append(("get_job", user_id))
        return await super().get_job(job_id, user_id=user_id)

    async def primary_resume(self, user_id):
        self.calls.append(("primary_resume", user_id))
        if user_id != DANA.id:
            return None
        return Resume(id=3, user_id=user_id, title="Dana platform CV", source="paste", content=RESUME_TEXT,
                      updated_at=NOW - timedelta(days=3), chars=len(RESUME_TEXT))

    async def list_resumes(self, user_id):
        self.calls.append(("list_resumes", user_id))
        if user_id != DANA.id:
            return []
        return [ResumeSummary(id=3, user_id=user_id, title="Dana platform CV", source="paste")]


TOOLS_TO_CALL = ("portal_overview", "search_jobs", "job_details", "my_resume", "pipeline_stats")


def tool_model(seen: list, *, tools=TOOLS_TO_CALL) -> FunctionModel:
    """Calls ``tools`` at once on the first request, then answers."""

    async def stream(messages, info: AgentInfo):
        seen.append((messages, info))
        last = messages[-1]
        if isinstance(last, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in last.parts):
            yield "Done."
            return
        offered = {t.name for t in info.function_tools}
        yield {i: DeltaToolCall(name=name, json_args=f'{{"job_id": {JOB.id}}}' if name == "job_details" else "{}",
                                tool_call_id=f"call-{i}") for i, name in enumerate(n for n in tools if n in offered)}

    return FunctionModel(stream_function=stream)


def tool_returns(seen: list) -> dict[str, str]:
    last = seen[-1][0][-1]
    return {p.tool_name: str(p.content) for p in last.parts if isinstance(p, ToolReturnPart)}


async def run_turn(user: User | None, database: StubDB, *, text="hello there, anything I should know?",
                   model: str | None = None, **settings) -> tuple[list, list]:
    seen: list = []
    settings = make_settings(router_mode="rules", **settings)
    advisor = Advisor(settings, models={"chat": tool_model(seen)})
    turn = ChatTurnRequest(messages=[{"role": "user", "content": text}], model=model)
    events = [e async for e in advisor.stream_turn(turn, database, FakeRuns(), user=user)]
    return events, seen


# --- user threading -----------------------------------------------------------------------


async def test_tools_read_the_signed_in_users_rows():
    database = RecordingDB()
    events, seen = await run_turn(DANA, database)
    assert [type(e).__name__ for e in events][-2:] == ["FinalEvent", "DoneEvent"]
    per_user = list(database.calls)
    assert per_user and all(uid == DANA.id for _, uid in per_user), per_user
    assert {name for name, _ in per_user} >= {"overview", "job_facets", "list_jobs", "get_job", "primary_resume"}


async def test_local_operator_keeps_the_shared_view():
    database = RecordingDB()
    await run_turn(None, database)
    assert all(uid in (None, 0) for _, uid in database.calls)
    assert ("primary_resume", 0) in database.calls  # the local account's resume


# --- my_resume -----------------------------------------------------------------------------


async def test_my_resume_with_a_resume():
    _, seen = await run_turn(DANA, RecordingDB())
    text = tool_returns(seen)["my_resume"]
    assert "Dana platform CV" in text and "source: paste" in text and "3d ago" in text
    assert "Kubernetes, Terraform" in text and "(clipped)" in text and len(text) <= 4000


async def test_my_resume_without_a_resume_says_where_to_add_one():
    _, seen = await run_turn(ERIK, RecordingDB())
    text = tool_returns(seen)["my_resume"]
    assert "no resume" in text and "/resume" in text and "/resume/build" in text
    assert "upload" in text and "paste" in text and '"Upload resume"' in text and "cannot receive files" not in text


# --- tool set per role and own numbers ----------------------------------------------------


@pytest.mark.parametrize("user", [DANA, ADMIN, None])
async def test_every_role_gets_the_same_tools(user):
    _, seen = await run_turn(user, RecordingDB())
    offered = {t.name for t in seen[0][1].function_tools}
    assert offered == {"portal_overview", "search_jobs", "job_details", "my_resume", "recent_runs", "pipeline_stats"}


async def test_regular_users_see_their_own_counts_and_no_corpus_internals():
    _, seen = await run_turn(DANA, RecordingDB())
    returns = tool_returns(seen)
    for name in ("portal_overview", "pipeline_stats"):
        assert "shared by every account" in returns[name] and "own triage" in returns[name]
        assert "Job sites:" not in returns[name]
    assert '"overview"' not in seen[0][1].instructions


# --- model choice -----------------------------------------------------------------------


def test_regular_users_fall_back_to_the_default_for_unlisted_models():
    settings = make_settings()
    listed = settings.available_models()[-1].id
    assert turn_model(settings, "openrouter:some/custom-model", DANA) == settings.default_model
    assert turn_model(settings, listed, DANA) == listed
    assert turn_model(settings, "openrouter:some/custom-model", ADMIN) == "openrouter:some/custom-model"
    with pytest.raises(ValueError):
        turn_model(make_settings(allow_custom_models=False), "openrouter:some/custom-model", ADMIN)


async def test_models_endpoint_hides_custom_ids_and_catalogue_from_regular_users(monkeypatch):
    async def fake_catalogue(http=None):
        return [{"id": "openrouter:x/y", "label": "x/y", "provider": "openrouter"}]

    monkeypatch.setattr(routes_chat, "openrouter_catalogue", fake_catalogue)
    bodies = {}
    for user in (DANA, ADMIN):
        monkeypatch.setattr(routes_chat, "current_user", lambda request, user=user: user)
        async with client_for(None, settings=make_settings()) as client:
            bodies[user.account_name] = (await client.get("/api/models?live=1")).json()
    assert bodies["dana"]["custom_allowed"] is False and "catalogue" not in bodies["dana"]
    assert bodies["root"]["custom_allowed"] is True and bodies["root"]["catalogue"]


# --- runs started from chat --------------------------------------------------------------


async def test_runs_started_from_chat_carry_the_account():
    runs = FakeRuns()
    action = await plan_action(RouteDecision(intent="stats", router="rules"), database=RecordingDB(), runs=runs,
                               settings=make_settings(), model_id="openrouter:openrouter/free", text="stats", user=DANA)
    assert isinstance(action, StatsAction) and action.run is not None
    assert runs.requests[0].user_id == DANA.id and runs.requests[0].requested_by == "dana"


async def test_local_operator_runs_keep_the_defaults():
    runs = FakeRuns()
    await plan_action(RouteDecision(intent="stats", router="rules"), database=RecordingDB(), runs=runs,
                      settings=make_settings(), model_id="openrouter:openrouter/free", text="stats")
    assert runs.requests[0].user_id is None and runs.requests[0].requested_by == "controller"


# --- the prompt ---------------------------------------------------------------------------


async def test_prompt_maps_the_real_product_and_forbids_invented_ui():
    _, seen = await run_turn(DANA, RecordingDB())
    prompt = seen[0][1].instructions
    for needle in ("Jobs:", "Explore (/scout):", "Resume (/resume)", "/resume/build",
                   '"Upload resume" button', "drop a resume file", "never read a file yourself",
                   "does not submit applications",
                   "A row's buttons are View, Applied and Not for me", '"I applied"', '"Open posting"',
                   "apply on the employer's site", 'named "Apply"', 'never "unreviewed"', "never mention them", "Steer the journey",
                   "Never describe a button, panel, page, upload control or feature that is not in the list above",
                   "Never promise to report back", "the panel next to this chat", "below the chat on a phone",
                   "Call my_resume before advice", "this user's own", "signed in as dana",
                   "Never ask them to report back", "then ask me again", "**bold** for a few key words"):
        assert needle in prompt, needle
    assert "the operator's job search" not in prompt and "admin of the portal" not in prompt
    for gone in ("Tailor my resume", "scout", "PDF run", "tailored"):
        assert gone not in prompt.replace("Explore (/scout)", ""), gone
    assert "Confirm on the right" not in prompt and "cannot receive files" not in prompt
    # With R3_JOB_POSTINGS=teaser (OQ-6), nothing in a regular account's brief offers the posting or its link.
    _, seen = await run_turn(DANA, RecordingDB(), job_postings="teaser")
    prompt = seen[0][1].instructions
    for needle in ('"About this role"', "does not show the original posting",
                   "Never give, guess or look up a posting's link"):
        assert needle in prompt, needle
    assert '"Open posting"' not in prompt and "link to the employer" not in prompt


async def test_prompt_keeps_operator_words_out_of_a_regular_users_chat():
    _, dana = await run_turn(DANA, RecordingDB())
    assert 'never "unreviewed"' in dana[0][1].instructions
    assert "never mention them" in dana[0][1].instructions
    _, root = await run_turn(ADMIN, RecordingDB())
    assert "Wording for this account" not in root[0][1].instructions
    assert "admin of the portal" in root[0][1].instructions


async def test_prompt_states_the_current_resume_status():
    _, seen = await run_turn(DANA, RecordingDB())
    assert "already has a resume in R3 (Dana platform CV" in seen[0][1].instructions
    assert "Do not tell them to upload one." in seen[0][1].instructions
    _, seen = await run_turn(ERIK, RecordingDB())
    assert "has no resume in R3 yet" in seen[0][1].instructions
    assert "Do not tell them to upload one." not in seen[0][1].instructions


async def test_admins_get_the_operator_paragraph():
    _, seen = await run_turn(ADMIN, RecordingDB())
    assert "admin of the portal" in seen[0][1].instructions
    assert '"Open posting"' in seen[0][1].instructions  # admins still read the posting itself


# --- the route end to end, two users -----------------------------------------------------


async def test_chat_route_threads_each_signed_in_user(monkeypatch):
    users = {"dana": DANA, "erik": ERIK}
    monkeypatch.setattr(routes_chat, "current_user", lambda request: users[request.headers["x-test-user"]])
    seen: list = []
    database = RecordingDB()
    settings = make_settings(router_mode="rules")
    advisor = Advisor(settings, models={"chat": tool_model(seen, tools=("my_resume", "search_jobs"))})
    replies = {}
    async with client_for(advisor, settings=settings, database=database) as client:
        for name in ("dana", "erik"):
            database.calls.clear()
            response = await client.post("/api/chat/stream", headers={"x-test-user": name},
                                         json={"messages": [{"role": "user", "content": "how should I position myself?"}],
                                               "model": "openrouter:some/custom-model"})
            assert response.status_code == 200 and "event: done" in response.text
            assert database.calls and all(uid == users[name].id for _, uid in database.calls), database.calls
            replies[name] = tool_returns(seen)["my_resume"]
    assert "Dana platform CV" in replies["dana"] and "no resume" in replies["erik"]


# --- the job teaser (OQ-6) in the chat ------------------------------------------------------


def test_named_jobs_narration_sends_a_seeker_for_the_overview_not_links():
    from r3.agent.advisor import narration
    from r3.agent.contracts import JobCard, JobsAction

    action = JobsAction(title="#7 Senior ML Engineer", jobs=[JobCard.from_job(JOB, full=False)], total=1)
    decision = RouteDecision(intent="jobs", router="rules")
    seeker = narration(decision, action, admin=False)
    assert "job_details" in seeker and "links" not in seeker and "description" not in seeker
    assert "links" in narration(decision, action, admin=True)


async def test_prompt_gives_explore_a_path_a_seeker_can_follow():
    _, seen = await run_turn(DANA, RecordingDB())
    assert "Explore (/scout)" in seen[0][1].instructions


# --- L13, L16, OQ-7: pricing, claims, links ------------------------------------------------------


async def test_prompt_rules_on_pricing_claims_and_links():
    _, seen = await run_turn(DANA, RecordingDB())
    prompt = seen[0][1].instructions
    for needle in ("Pricing has not been announced", "never invent a price", "no more 403s",
                   "unless the panel started that run in this turn", "[Position at Company](/jobs/<id>)",
                   "[Build your resume](/resume/build)", "never a bare path"):
        assert needle in prompt, needle
    assert "#<id> Position at Company" not in prompt and "#id Position at Company" not in prompt


def _narrate(reason: str, admin: bool = False) -> str:
    from r3.agent.advisor import narration
    from r3.agent.contracts import AdviceAction

    return narration(RouteDecision(intent="advice", router="rules", reason=reason), AdviceAction(), admin=admin)


def test_resume_help_narration_gives_the_builder_link_and_the_upload_button():
    from r3.agent.router import RESUME_HELP

    text = _narrate(RESUME_HELP)
    for needle in ("[Build your resume](/resume/build)", '"Upload resume" button', "[Resume](/resume)",
                   "Never say that you or this chat cannot help", "describe their background"):
        assert needle in text, needle


def test_the_product_brief_offers_a_resume_from_a_described_background():
    from r3.agent.advisor import LIMITS, PRODUCT

    assert "describe their background" in PRODUCT
    assert "never ask them to paste a whole resume" not in LIMITS


def test_pricing_narration():
    from r3.agent.router import PRICING

    assert "Pricing has not been announced" in _narrate(PRICING) and "who pays" in _narrate(PRICING)
    assert "Pricing" not in _narrate("asks for advice or an explanation")


@pytest.mark.parametrize("mode", ["auto", "advice"])
async def test_the_resume_starter_gets_the_builder_link_in_either_mode(mode):
    from r3.agent.router import RESUME_HELP

    seen: list = []

    async def stream(messages, info: AgentInfo):
        seen.append(info)
        yield "Here is how."

    advisor = Advisor(make_settings(router_mode="rules"), models={"chat": FunctionModel(stream_function=stream)})
    turn = ChatTurnRequest(messages=[{"role": "user", "content": "Help me build my resume"}], mode=mode)
    events = [e async for e in advisor.stream_turn(turn, RecordingDB(), FakeRuns(), user=ERIK)]
    assert events[0].decision.intent == "advice" and events[0].decision.reason == RESUME_HELP
    assert events[1].action.kind == "advice"
    assert "[Build your resume](/resume/build)" in seen[0].instructions
    assert '"Upload resume" button' in seen[0].instructions


async def test_job_tools_lead_with_the_job_link_not_an_id():
    _, seen = await run_turn(DANA, RecordingDB())
    link = f"[{JOB.position} at {JOB.company}](/jobs/{JOB.id})"
    returns = tool_returns(seen)
    for name in ("search_jobs", "job_details"):
        assert link in returns[name] and f"#{JOB.id}" not in returns[name], (name, returns[name])


def test_job_narrations_name_jobs_as_links_without_ids():
    from r3.agent.advisor import narration
    from r3.agent.contracts import JobCard, JobsAction

    card = JobCard.from_job(JOB, full=False)
    link = f"[{JOB.position} at {JOB.company}](/jobs/{JOB.id})"
    listed = JobsAction(title="1 new job", filter_query="?status=new", jobs=[card], total=1)
    named = JobsAction(title=f"#{JOB.id} {JOB.position}; not found: #999", jobs=[card], total=1)
    decision = RouteDecision(intent="jobs", router="rules")
    for action in (listed, named):
        text = narration(decision, action, admin=False)
        assert link in text, text
        assert f"#{JOB.id} " not in text and "#id" not in text, text
    assert "Not found: #999" in narration(decision, named, admin=False)
