"""A resume made from what the person wrote in the Assistant chat: the
model extraction and its guards, when the chat sees a background, the chat
turn that hands it to the page, and ``POST /api/resume/from-text``.
In-memory stub and fake models (no network)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from r3.agent.contracts import ChatMessage, ChatTurnRequest
from r3.agent.router import resume_background
from r3.agent.service import Advisor
from r3.resume.builder import (
    Contact,
    Job,
    LLMInterviewer,
    ResumeDraft,
    ScriptedInterviewer,
)
from r3.web import create_app, routes_resume
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_agent_users import ERIK, RecordingDB
from tests.test_resume import ALICE, BOB, ResumeStub, as_user
from tests.test_resume_chat_api import CountingParser
from tests.test_resume_parse import CLASSIC

BACKGROUND = ("I'm Lee Park, a backend developer in Austin. Since 2021 I've worked at Stripe on the payments API in "
              "Go and Python, and cut p99 latency by 40%. Before that I was a software engineer at Initech building "
              "internal tools with Django. I know Go, Python, PostgreSQL, Kafka and Kubernetes. BS in Computer "
              "Science, UT Austin 2018. lee.park@example.com")
ASK = "make me a resume from this: " + BACKGROUND

LEE = ResumeDraft(
    full_name="Lee Park", headline="Backend Developer",
    contact=Contact(email="lee.park@example.com", location="Austin"),
    experience=[Job(title="Backend Developer", company="Stripe", start="2021", end="Present",
                    bullets=["Built the payments API in Go and Python", "Cut p99 latency by 40%"]),
                Job(title="Software Engineer", company="Initech", bullets=["Built internal tools with Django"])],
    skills=["Go", "Python", "PostgreSQL", "Kafka", "Kubernetes"],
    education=["BS in Computer Science, UT Austin, 2018"])


def model_returning(draft: ResumeDraft | dict) -> FunctionModel:
    payload = draft.model_dump() if isinstance(draft, ResumeDraft) else draft

    def respond(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps(payload))])
    return FunctionModel(respond)


# --- the extraction ----------------------------------------------------------------------

async def test_a_background_is_extracted_into_a_draft():
    draft = await LLMInterviewer(model_returning(LEE)).extract(BACKGROUND)
    assert draft.full_name == "Lee Park" and [j.company for j in draft.experience] == ["Stripe", "Initech"]
    assert draft.contact.email == "lee.park@example.com"


async def test_an_invented_number_rejects_the_extraction():
    made_up = LEE.model_copy(deep=True)
    made_up.experience[0].bullets.append("Served 3 million merchants")
    with pytest.raises(ValueError, match="numbers"):
        await LLMInterviewer(model_returning(made_up)).extract(BACKGROUND)


async def test_an_invented_employer_rejects_the_extraction():
    made_up = LEE.model_copy(deep=True)
    made_up.experience[1].company = "Globex"
    with pytest.raises(ValueError, match="Globex"):
        await LLMInterviewer(model_returning(made_up)).extract(BACKGROUND)


async def test_skills_the_person_never_wrote_are_dropped():
    padded = LEE.model_copy(deep=True)
    padded.skills += ["Terraform", "kubernetes"]
    draft = await LLMInterviewer(model_returning(padded)).extract(BACKGROUND)
    assert draft.skills == ["Go", "Python", "PostgreSQL", "Kafka", "Kubernetes"]


async def test_a_slow_extraction_times_out():
    async def slow(messages, info):
        await asyncio.sleep(1)
        return ModelResponse(parts=[TextPart("{}")])

    with pytest.raises(TimeoutError):
        await LLMInterviewer(FunctionModel(slow), timeout_s=0.01).extract(BACKGROUND)


# --- when the chat sees a background --------------------------------------------------------

def test_a_request_with_a_background_carries_it():
    assert resume_background(ASK) == ASK
    assert resume_background(BACKGROUND + " Can you make me a resume?") == BACKGROUND + " Can you make me a resume?"


def test_a_bare_request_has_no_background():
    assert resume_background("make me a resume") == ""
    assert resume_background("help me build my resume please") == ""


def test_a_short_request_after_a_pasted_background_uses_that_message():
    history = [ChatMessage(role="user", content=BACKGROUND), ChatMessage(role="assistant", content="Nice background.")]
    assert resume_background("make me a resume from this", history) == BACKGROUND
    # Without a pointer back ("this", "that", "above") an older message is not taken.
    assert resume_background("make me a resume", history) == ""


# --- the chat turn ------------------------------------------------------------------------

async def test_a_resume_request_with_a_background_hands_it_to_the_page_without_a_model_call():
    async def never(messages, info):
        raise AssertionError("the chat model must not be called")
        yield ""  # pragma: no cover

    advisor = Advisor(make_settings(router_mode="rules"), models={"chat": FunctionModel(stream_function=never)})
    turn = ChatTurnRequest(messages=[{"role": "user", "content": ASK}])
    events = [e async for e in advisor.stream_turn(turn, RecordingDB(), FakeRuns(), user=ERIK)]
    kinds = [getattr(e, "type", "") for e in events]
    assert kinds == ["route", "action", "final", "done"], kinds
    assert events[1].action.kind == "resume_text" and events[1].action.text == ASK
    assert "making a resume" in events[2].text.lower()


async def test_advice_only_mode_never_makes_a_resume():
    async def stream(messages, info):
        yield "Here is how."

    advisor = Advisor(make_settings(router_mode="rules"), models={"chat": FunctionModel(stream_function=stream)})
    turn = ChatTurnRequest(messages=[{"role": "user", "content": ASK}], mode="advice")
    events = [e async for e in advisor.stream_turn(turn, RecordingDB(), FakeRuns(), user=ERIK)]
    assert events[1].action.kind == "advice"


# --- POST /api/resume/from-text ---------------------------------------------------------------

class FakeInterviewer(ScriptedInterviewer):
    model_id = "openrouter:openai/gpt-6-luna"

    def __init__(self, draft: ResumeDraft | None = LEE, error: Exception | None = None) -> None:
        self.draft, self.error, self.calls = draft, error, []

    async def extract(self, text: str) -> ResumeDraft:
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return self.draft


@pytest.fixture
def stub():
    return ResumeStub()


@pytest.fixture
async def make_client(stub):
    async with contextlib.AsyncExitStack() as stack:
        async def make(interviewer=None, user=ALICE) -> httpx.AsyncClient:
            app = create_app(database=stub, resume_parser=CountingParser(),
                             resume_interviewer=interviewer if interviewer is not None else FakeInterviewer())
            await stack.enter_async_context(app.router.lifespan_context(app))
            return await stack.enter_async_context(
                httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, user)), base_url="http://test"))

        yield make


async def test_a_background_becomes_a_saved_resume_with_follow_ups(make_client, stub):
    interviewer = FakeInterviewer()
    r = await (await make_client(interviewer)).post("/api/resume/from-text", json={"text": ASK})
    assert r.status_code == 200, r.text
    card = r.json()
    assert card["name"] == "Lee Park" and [x["company"] for x in card["roles"]] == ["Stripe", "Initech"]
    assert card["primary"] is True and card["engine"] == "background"
    assert card["topic"] == "dates" and "Initech" in card["question"]  # Initech has no dates
    row = stub.resumes[card["resume_id"]]
    assert row["source"] == "chat" and row["content"].startswith("Lee Park\nBackend Developer")
    assert row["structure"]["model"] == "openrouter:openai/gpt-6-luna" and row["structure"]["source"] == ASK
    assert interviewer.calls == [ASK]


async def test_a_pasted_resume_is_read_like_an_upload(make_client, stub):
    interviewer = FakeInterviewer()
    card = (await (await make_client(interviewer)).post("/api/resume/from-text", json={"text": CLASSIC})).json()
    assert card["name"] == "Jane Doe" and card["engine"] == "heuristic" and card["faithful"] is True
    assert interviewer.calls == []  # resume-shaped text needs no extraction


@pytest.mark.parametrize(("interviewer", "status", "words"), [
    (FakeInterviewer(error=ValueError("model added numbers")), 422, "couldn't turn"),
    (FakeInterviewer(draft=ResumeDraft()), 422, "background"),
    (ScriptedInterviewer(), 422, "Build your resume"),
])
async def test_a_background_that_cannot_be_read_saves_nothing(make_client, stub, interviewer, status, words):
    r = await (await make_client(interviewer)).post("/api/resume/from-text", json={"text": ASK})
    assert r.status_code == status and words in r.json()["detail"], r.text
    assert stub.resumes == {}


async def test_from_text_checks_length_and_the_hourly_limit(make_client, stub):
    c = await make_client()
    r = await c.post("/api/resume/from-text", json={"text": "too short"})
    assert r.status_code == 400 and "characters" in r.json()["detail"]
    stub.clock = datetime.now(UTC) - timedelta(minutes=30)
    for i in range(routes_resume.UPLOADS_PER_HOUR):
        await stub.create_resume(ALICE.id, title=f"cv{i}", source="chat" if i % 2 else "upload", content=CLASSIC)
    r = await c.post("/api/resume/from-text", json={"text": ASK})
    assert r.status_code == 429 and "hour" in r.json()["detail"]
    r = await (await make_client(user=BOB)).post("/api/resume/from-text", json={"text": ASK})
    assert r.status_code == 200
