"""The resume builder: the scripted and LLM interviewers, rendering and
progress, and the /resume/build routes over an in-memory stand-in for
:class:`r3.db.resume_drafts.ResumeDraftQueries`. No network, no database."""

import asyncio
import html
import json

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from r3.db.resume_drafts import StoredDraft
from r3.resume.builder import (
    DONE_QUESTION, QUESTIONS, BuilderTurn, ChatEntry, Contact, Job, LLMInterviewer, ResumeDraft,
    ScriptedInterviewer, completeness, dates_question, invented_numbers, render_resume_text,
)
from r3.web import create_app
from tests.test_resume import ALICE, BOB, PREFIX, XSS, ResumeStub, as_user

ANSWERS = [
    "Senior Data Analyst",
    "Jane Doe",
    "jane@example.com, +1 555 010 2000, Berlin, linkedin.com/in/janedoe",
    "Data Analyst, Acme GmbH, Berlin, 2021, present",
    "Cut weekly report time from 2 days to 2 hours\nBuilt the churn dashboard used by 40 account managers",
    "Junior Analyst, Beta AG, Hamburg, 2018, 2021",
    "Cleaned CRM data for 3 sales teams",
    "no",
    "SQL, Python, Tableau",
    "BSc Statistics, University of Hamburg, 2018",
    "skip",
    "German (native), English (fluent)",
    "Analyst who turns messy data into decisions.",
]

EXPECTED = """\
Jane Doe
Senior Data Analyst
jane@example.com | +1 555 010 2000 | Berlin | linkedin.com/in/janedoe

SUMMARY
Analyst who turns messy data into decisions.

EXPERIENCE
Data Analyst — Acme GmbH, Berlin
2021 – present
- Cut weekly report time from 2 days to 2 hours
- Built the churn dashboard used by 40 account managers

Junior Analyst — Beta AG, Hamburg
2018 – 2021
- Cleaned CRM data for 3 sales teams

SKILLS
SQL, Python, Tableau

EDUCATION
- BSc Statistics, University of Hamburg, 2018

ADDITIONAL
- German (native), English (fluent)"""


async def run_script(interviewer, answers):
    draft, history = ResumeDraft(), []
    opening = interviewer.opening(draft)
    history.append(ChatEntry(role="assistant", text=opening.question, topic=opening.topic))
    turn = None
    for answer in answers:
        turn = await interviewer.next_turn(draft, history, answer)
        draft = turn.draft
        history += [ChatEntry(role="user", text=answer),
                    ChatEntry(role="assistant", text=turn.question, topic=turn.topic, done=turn.done)]
    return turn, history


# ---------------------------------------------------------------- scripted

async def test_scripted_interview_end_to_end_builds_a_complete_resume():
    turn, _ = await run_script(ScriptedInterviewer(), ANSWERS)
    assert turn.done and turn.question == DONE_QUESTION
    assert all(ok for _, ok in completeness(turn.draft))
    assert render_resume_text(turn.draft) == EXPECTED


async def test_scripted_skip_moves_on_and_leaves_sections_out():
    answers = ["Nurse", "Sam Lee", "sam@example.org", "I don't have one", "Patient care", "skip", "skip",
               "nothing", "Caring and calm.", "skip"]
    turn, history = await run_script(ScriptedInterviewer(), answers)
    # "I don't have one" at the job question jumps straight to skills.
    assert history[8].text == QUESTIONS["skills"]
    # No links were given, so the interview asks once (and explains why) before done.
    assert "verifiable" in history[-3].text and "LinkedIn" in history[-3].text
    assert turn.done and turn.draft.experience == [] and turn.draft.education == []
    assert turn.draft.skipped == ["links"]
    text = render_resume_text(turn.draft)
    assert "EXPERIENCE" not in text and "EDUCATION" not in text and "CERTIFICATIONS" not in text
    assert text.startswith("Sam Lee\nNurse\nsam@example.org\n\nSUMMARY")


async def test_scripted_presses_for_dates_and_a_link_before_done():
    answers = ["Dev", "Kim Park", "k@x.io", "Dev, Acme, Berlin", "Shipped the API",
               "no", "Go, Python", "skip", "skip", "no", "Builder who ships.", "2019, 2022", "github.com/kim"]
    turn, history = await run_script(ScriptedInterviewer(), answers)
    # Two required follow-ups, in order: dates for the dateless role, then a link.
    assert history[-5].text.startswith("When did you start and end at Dev?")
    assert "verifiable" in history[-3].text and "LinkedIn" in history[-3].text
    assert turn.done and turn.draft.experience[0].start == "2019" and turn.draft.experience[0].end == "2022"
    assert turn.draft.contact.links == ["github.com/kim"]
    got = dict(completeness(turn.draft))
    assert got["Experience"] and got["Profile link"]


async def test_done_state_typing_fixes_a_missing_requirement_instead_of_extras():
    iv = ScriptedInterviewer()
    answers = ["Dev", "Kim Park", "k@x.io", "Dev, Acme, Berlin, 2019, 2022", "Shipped the API",
               "no", "Go, Python", "skip", "skip", "no", "Builder who ships.", "skip"]
    turn, history = await run_script(iv, answers)  # links skipped at the follow-up: done with the requirement open
    assert turn.done and turn.draft.skipped == ["links"] and turn.draft.contact.links == []
    nxt = await iv.next_turn(turn.draft, history, "github.com/kim")
    assert nxt.done and nxt.draft.contact.links == ["github.com/kim"] and nxt.draft.extras == []
    more = await iv.next_turn(nxt.draft, history, "Fluent in German")
    assert more.draft.extras == ["Fluent in German"]


async def test_a_lone_end_date_completes_a_role_that_has_its_start():
    # "Dev, Acme, Berlin, 2019" left the end open; the natural reply "present"
    # must fill it (M2), not silently re-ask the same question.
    answers = ["Dev", "Kim Park", "k@x.io, github.com/kim", "Dev, Acme, Berlin, 2019", "Did things",
               "no", "Go", "skip", "skip", "no", "Did.", "present"]
    turn, _ = await run_script(ScriptedInterviewer(), answers)
    assert turn.done and turn.draft.experience[0].start == "2019" and turn.draft.experience[0].end == "present"


async def test_a_guessed_contact_filing_never_reasks_the_job_question():
    # LLM failed right after its own question; the guess files a dates answer
    # under "contact" — the next question must be the dates follow-up, not
    # "tell me about your most recent job" again (which would add a duplicate).
    def boom(messages, info):
        raise RuntimeError("provider down")
    draft = ResumeDraft(headline="Dev", full_name="Kim", contact=Contact(email="k@x.io"),
                        experience=[Job(title="Dev", company="Acme", start="2019", bullets=["Did things"])],
                        skills=["Go"], education=["BSc"], certifications=["AWS"], summary="Did.")
    history = [ChatEntry(role="assistant", text="When did you start at Acme?")]
    turn = await LLMInterviewer(FunctionModel(boom)).next_turn(draft, history, "2022")
    assert len(turn.draft.experience) == 1
    assert turn.question.startswith("When did you start and end at Dev?")


async def test_scripted_pasted_block_keeps_every_word_and_reasks():
    block = "\n".join(["Jane Doe", "jane@example.com", "https://github.com/jane"] + [f"line {i}" for i in range(12)])
    turn, _ = await run_script(ScriptedInterviewer(), [block])
    assert turn.question == QUESTIONS["role"] and turn.topic == "role"
    assert turn.draft.contact.email == "jane@example.com"
    assert turn.draft.contact.links == ["https://github.com/jane"]
    assert turn.draft.extras == [block] and "kept it whole" in turn.note


# ---------------------------------------------------------------- units

def test_render_empty_and_partial_drafts():
    assert render_resume_text(ResumeDraft()) == ""
    draft = ResumeDraft(full_name="  Ann  Smith ", skills=["Go", " ", "Rust"],
                        experience=[Job(company="Initech", bullets=["Shipped  v2"])])
    assert render_resume_text(draft) == "Ann Smith\n\nEXPERIENCE\nInitech\n- Shipped v2\n\nSKILLS\nGo, Rust"
    assert render_resume_text(draft) == render_resume_text(draft.model_copy(deep=True))


def test_completeness_counts_each_section():
    assert [ok for _, ok in completeness(ResumeDraft())] == [False] * 8
    draft = ResumeDraft(headline="PM", full_name="Al", contact=Contact(phone="123"),
                        certifications=["PMP"], experience=[Job(title="PM", start="2020", end="2024")])
    assert dict(completeness(draft)) == {
        "Target role": True, "Name and contact": True, "Experience": True, "Achievements": False,
        "Skills": False, "Education or certifications": True, "Summary": False, "Profile link": False}
    # Experience without dates does not count (OQ-3), and a link is its own section (OQ-2).
    assert completeness(ResumeDraft(experience=[Job(title="PM")]))[2] == ("Experience", False)
    assert completeness(ResumeDraft(contact=Contact(links=["github.com/a"])))[7] == ("Profile link", True)


def test_invented_numbers_flags_only_new_unsaid_numbers():
    before = ResumeDraft(experience=[Job(title="Dev", start="2019")])
    after = ResumeDraft(experience=[Job(title="Dev", start="2019", bullets=["Cut costs 30%", "Led 5 people"])])
    assert invented_numbers(before, after, "I led 5 people") == {"30"}


# ---------------------------------------------------------------- LLM

def model_returning(payload: dict) -> FunctionModel:
    def respond(messages, info):
        return ModelResponse(parts=[TextPart(json.dumps(payload))])
    return FunctionModel(respond)


async def test_llm_turn_is_merged_into_the_draft():
    payload = {"draft": ResumeDraft(headline="Backend Engineer", full_name="Kim Park").model_dump(),
               "question": "What is your email?", "done": False}
    history = [ChatEntry(role="assistant", text=QUESTIONS["role"], topic="role")]
    turn = await LLMInterviewer(model_returning(payload)).next_turn(
        ResumeDraft(), history, "Backend engineer. I'm Kim Park.")
    assert turn.draft.headline == "Backend Engineer" and turn.draft.full_name == "Kim Park"
    assert turn.question == "What is your email?" and turn.note == "" and not turn.done


async def test_llm_failure_falls_back_to_the_script_without_losing_the_answer():
    def boom(messages, info):
        raise RuntimeError("provider down")
    history = [ChatEntry(role="assistant", text=QUESTIONS["role"], topic="role")]
    turn = await LLMInterviewer(FunctionModel(boom)).next_turn(ResumeDraft(), history, "Product Designer")
    assert turn.draft.headline == "Product Designer"
    assert turn.question == QUESTIONS["name"] and "filed your answer as typed" in turn.note


async def test_llm_invented_number_is_rejected_and_answer_filed_verbatim():
    job = Job(title="Dev", company="Acme", bullets=["Raised revenue 25%"])
    payload = {"draft": ResumeDraft(experience=[job]).model_dump(), "question": "Next?", "done": False}
    # An LLM question (no topic): the fallback files into the first missing section.
    history = [ChatEntry(role="assistant", text="What did you do at Acme?")]
    draft = ResumeDraft(headline="Dev", full_name="Al", contact=Contact(email="a@b.co"),
                        experience=[Job(title="Dev", company="Acme")])
    turn = await LLMInterviewer(model_returning(payload)).next_turn(draft, history, "I raised revenue a lot")
    assert turn.draft.experience[0].bullets == ["I raised revenue a lot"]
    assert "25" not in turn.draft.model_dump_json()


async def test_llm_timeout_falls_back():
    never = asyncio.Event()

    async def slow(messages, info):
        await never.wait()
    turn = await LLMInterviewer(FunctionModel(slow), timeout_s=0.01).next_turn(
        ResumeDraft(), [ChatEntry(role="assistant", text=QUESTIONS["role"], topic="role")], "Chef")
    assert turn.draft.headline == "Chef" and "took too long" in turn.note


# ---------------------------------------------------------------- routes

class DraftStub(ResumeStub):
    """Adds ResumeDraftQueries, keyed by user_id like the real table."""

    def __init__(self) -> None:
        super().__init__()
        self.drafts: dict[int, StoredDraft] = {}

    async def ensure_resume_draft_tables(self) -> None:
        pass

    async def get_resume_draft(self, user_id):
        return self.drafts.get(user_id)

    async def save_resume_draft(self, user_id, *, draft, history):
        # Round-trip through JSON like jsonb does.
        self.drafts[user_id] = StoredDraft(user_id=user_id, draft=json.loads(json.dumps(draft)),
                                           history=json.loads(json.dumps(history)))

    async def delete_resume_draft(self, user_id):
        return self.drafts.pop(user_id, None) is not None


class Raising:
    def opening(self, draft):
        return ScriptedInterviewer().opening(draft)

    async def next_turn(self, draft, history, answer) -> BuilderTurn:
        raise RuntimeError("boom")


@pytest.fixture
def stub():
    return DraftStub()


async def clients_for(app, *users):
    return [httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, u)), base_url="http://test")
            for u in users]


@pytest.fixture
async def app_ctx(stub):
    app = create_app(database=stub, resume_interviewer=ScriptedInterviewer())
    async with app.router.lifespan_context(app):
        yield app


@pytest.fixture
async def alice(app_ctx):
    (c,) = await clients_for(app_ctx, ALICE)
    yield c
    await c.aclose()


async def answer(client, text, prefix=""):
    return await client.post(f"{prefix}/resume/build/answer", json={"answer": text})


async def test_page_shows_first_question_and_scripted_notice(alice):
    page = await alice.get("/resume/build")
    assert page.status_code == 200
    assert html.unescape(page.text).count(QUESTIONS["role"]) == 1 and "No AI model is configured" in page.text
    assert "0 of 8" in page.text and 'src="/static/resume_build.js?v=' in page.text


async def test_build_route_is_not_swallowed_by_resume_id(app_ctx, alice):
    # Starlette matches in registration order, so the builder's router must
    # come first; included routers may be wrapped, so flatten them.
    def paths(routes):
        for r in routes:
            inner = getattr(r, "original_router", None)
            yield from paths(inner.routes) if inner is not None else [getattr(r, "path", "")]
    order = list(paths(app_ctx.routes))
    assert order.index("/resume/build") < order.index("/resume/{resume_id:int}")
    page = await alice.get("/resume/build")
    assert page.status_code == 200 and "Resume interview" in page.text


async def test_answer_persists_and_progress_advances(alice, stub):
    res = await answer(alice, "Senior Data Analyst")
    assert res.status_code == 200
    body = res.json()
    assert body["question"] == QUESTIONS["name"] and body["progress"]["done"] == 1
    assert "Senior Data Analyst" in body["draft_html"] and "1 of 8" in body["progress_html"]
    assert stub.drafts[ALICE.id].draft["headline"] == "Senior Data Analyst"
    assert [e["role"] for e in stub.drafts[ALICE.id].history] == ["assistant", "user", "assistant"]
    # A reload restores the conversation and the preview.
    page = await alice.get("/resume/build")
    assert "Senior Data Analyst" in page.text and QUESTIONS["name"] in page.text and "1 of 8" in page.text


async def test_save_creates_builder_resume_and_clears_draft(alice, stub):
    for text in ANSWERS:
        assert (await answer(alice, text)).status_code == 200
    res = await alice.post("/resume/build/save", data={"title": ""})
    assert res.status_code == 303
    (row,) = stub.resumes.values()
    assert res.headers["location"] == f"/resume/{row['id']}"
    assert row["source"] == "builder" and row["user_id"] == ALICE.id and row["content"] == EXPECTED
    assert row["title"] == "Jane Doe — Senior Data Analyst"
    assert ALICE.id not in stub.drafts


async def test_thin_draft_is_refused(alice, stub):
    await answer(alice, "Chef")
    res = await alice.post("/resume/build/save", data={"title": "Mine"})
    assert res.status_code == 400 and "a resume needs at least 50" in res.text
    assert stub.resumes == {} and ALICE.id in stub.drafts


async def test_save_requires_a_link_and_dates_before_the_library(alice, stub):
    answers = ["Dev", "Kim Park", "k@x.io", "Dev, Acme, Berlin", "Shipped the API",
               "no", "Go, Python", "skip", "skip", "no", "Builder who ships."]
    for text in answers:
        assert (await answer(alice, text)).status_code == 200
    res = await alice.post("/resume/build/save", data={"title": "Mine"})
    assert res.status_code == 400 and "profile link" in res.text and "dates for: Dev" in res.text
    assert stub.resumes == {}
    # Typing the missing pieces in the interview fixes it; the save goes through.
    assert (await answer(alice, "2019, 2022")).status_code == 200
    assert (await answer(alice, "github.com/kim")).status_code == 200
    ok = await alice.post("/resume/build/save", data={"title": "Mine"})
    assert ok.status_code == 303 and stub.resumes


async def test_interviewer_instructions_carry_the_owner_requirements():
    from r3.resume.builder import INSTRUCTIONS
    for needle in ("start and end dates", "ask one short follow-up for just the dates", "never\n  invent or guess",
                   "LinkedIn, GitHub or a portfolio", "makes the resume\n  verifiable",
                   "steers the next step", "best matches are at the top", "as Applied"):
        assert needle in INSTRUCTIONS, needle


async def test_reset_starts_over(alice, stub):
    await answer(alice, "Chef")
    res = await alice.post("/resume/build/reset")
    assert res.status_code == 303 and res.headers["location"] == "/resume/build"
    assert ALICE.id not in stub.drafts
    assert "0 of 8" in (await alice.get("/resume/build")).text


async def test_answer_validation(alice, stub):
    assert (await answer(alice, "   ")).status_code == 400
    too_long = await answer(alice, "x" * 8001)
    assert too_long.status_code == 413 and "8,000" in too_long.json()["error"]
    assert stub.drafts == {}


async def test_failing_interviewer_returns_error_and_saves_nothing(stub):
    app = create_app(database=stub, resume_interviewer=Raising())
    async with app.router.lifespan_context(app):
        (c,) = await clients_for(app, ALICE)
        res = await answer(c, "Chef")
        await c.aclose()
    assert res.status_code == 502 and "back in the box" in res.json()["error"] and stub.drafts == {}


async def test_users_never_see_each_others_draft(app_ctx, stub):
    alice, bob = await clients_for(app_ctx, ALICE, BOB)
    await answer(alice, "Astronaut")
    assert "Astronaut" not in (await bob.get("/resume/build")).text
    await answer(bob, "Baker")
    await bob.post("/resume/build/reset")
    assert stub.drafts[ALICE.id].draft["headline"] == "Astronaut" and BOB.id not in stub.drafts
    await alice.aclose()
    await bob.aclose()


async def test_answer_is_escaped_in_preview(alice):
    body = (await answer(alice, XSS)).json()
    assert XSS not in body["draft_html"] and "&lt;script&gt;" in body["draft_html"]
    page = await alice.get("/resume/build")
    assert XSS not in page.text and "&lt;script&gt;alert" in page.text


async def test_root_path_prefix(stub):
    app = create_app(database=stub, root_path=PREFIX, resume_interviewer=ScriptedInterviewer())
    async with app.router.lifespan_context(app):
        (c,) = await clients_for(app, ALICE)
        page = await c.get(f"{PREFIX}/resume/build")
        assert page.status_code == 200
        assert f'action="{PREFIX}/resume/build/reset"' in page.text and f'data-base-path="{PREFIX}"' in page.text
        assert (await answer(c, "Chef", PREFIX)).status_code == 200
        reset = await c.post(f"{PREFIX}/resume/build/reset")
        assert reset.headers["location"] == f"{PREFIX}/resume/build"
        library = await c.get(f"{PREFIX}/resume")
        assert f'href="{PREFIX}/resume/build"' in library.text
        await c.aclose()


async def test_llm_fallback_keeps_a_multiline_answer_whole_after_a_model_question():
    def boom(messages, info):
        raise RuntimeError("provider down")
    block = "Software Engineer, Lumen, Porto, 2018-2021\n- Built the API gateway\nSkills: Go, Python"
    history = [ChatEntry(role="assistant", text="Tell me about earlier jobs?")]
    draft = ResumeDraft(headline="Dev", full_name="Kim", contact=Contact(email="k@x.io"),
                        experience=[Job(title="Dev", company="Tide", bullets=["Moved billing to Go"])])
    turn = await LLMInterviewer(FunctionModel(boom)).next_turn(draft, history, block)
    assert turn.draft.skills == [] and turn.draft.extras == [block]
