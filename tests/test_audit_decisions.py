"""The decisions in R3's audit trail: who changed what, what a run did, and
what was (and was not) sent anywhere. Each is one plain-sentence event."""

from __future__ import annotations

import pytest

from r3.events import default_log, trace_scope
from r3.runs.models import StatsParams
from tests.test_audit import AuditStub, _call
from tests.test_events import MemoryStore
from tests.test_routes import JOB
from tests.test_runs_local import make_runs, run_to_end, stub  # noqa: F401  (fixtures)


@pytest.fixture
def trail():
    """Audit events recorded outside a portal (runs, activities)."""
    store = MemoryStore()
    previous = default_log.store
    default_log.store = store
    yield store
    default_log.store = previous


async def _flushed(store):
    await default_log.flush()
    return store.rows


async def test_marking_a_job_applied_records_the_change_and_who_made_it():
    s = AuditStub()
    await _call(s, "POST", f"/jobs/{JOB.id}/status", data={"status": "applied", "redirect": "1"})
    ev = next(e for e in s.events if e.kind == "job.status")
    assert (ev.actor, ev.user_id, ev.subject) == ("user", 42, f"job:{JOB.id}")
    assert ev.data == {"from": "new", "to": "applied", "via": "job page"}
    assert ev.summary == f"You marked “{JOB.position} at {JOB.company}” applied"


async def test_a_run_records_its_start_and_its_end_in_one_trace(make_runs, trail):  # noqa: F811
    with trace_scope("a" * 32, user_id=5):
        await run_to_end(make_runs(), StatsParams(days=7), user_id=5)
    rows = await _flushed(trail)
    assert [e.kind for e in rows] == ["run.started", "run.finished"]
    assert all(e.trace_id == "a" * 32 for e in rows)  # the run stays in the trace that started it
    finished = rows[-1]
    assert finished.data["status"] == "completed" and finished.user_id == 5


async def test_a_run_start_names_its_kind_its_owner_and_its_parameters(make_runs, trail):  # noqa: F811
    await run_to_end(make_runs(), StatsParams(days=7), user_id=5, source="chat")
    started = next(e for e in await _flushed(trail) if e.kind == "run.started")
    assert started.user_id == 5 and started.subject.startswith("run:")
    assert started.data["kind"] == "stats" and started.data["params"]["days"] == 7
    assert started.data["source"] == "chat"


async def test_a_chat_turn_records_the_question_and_what_the_ai_decided():
    from r3.agent.contracts import RouteDecision
    from r3.agent.events import (
        FinalEvent,
        RouteEvent,
        ToolCallEvent,
        ToolResultEvent,
        Usage,
    )
    from tests.fakes import ScriptedAdvisor, test_settings

    advisor = ScriptedAdvisor(test_settings(), [
        RouteEvent(decision=RouteDecision(intent="advice", router="rules", reason="a question")),
        ToolCallEvent(name="job_details", args={"job_id": JOB.id}),
        ToolResultEvent(name="job_details", summary="DevOps Engineer at Northwind Labs"),
        FinalEvent(text="Northwind Labs fits your SRE background best.", model="openrouter/free",
                   usage=Usage(input_tokens=900, output_tokens=40)),
    ])
    s = AuditStub()
    r = await _call(s, "POST", "/api/chat/stream", app_kwargs={"advisor": advisor},
                    json={"messages": [{"role": "user", "content": "Which job fits me best?"}]})
    assert r.status_code == 200
    asked = next(e for e in s.events if e.kind == "chat.message")
    assert (asked.actor, asked.user_id) == ("user", 42)
    assert asked.data["text"] == "Which job fits me best?" and asked.summary == "You asked: Which job fits me best?"
    reply = next(e for e in s.events if e.kind == "chat.reply")
    assert (reply.actor, reply.user_id, reply.trace_id) == ("ai", 42, asked.trace_id)
    assert reply.data["text"] == "Northwind Labs fits your SRE background best."
    assert reply.data["model"] == "openrouter/free" and reply.data["usage"]["input_tokens"] == 900
    assert reply.data["route"]["intent"] == "advice"
    assert reply.data["tools"] == [{"name": "job_details", "args": {"job_id": JOB.id},
                                    "result": "DevOps Engineer at Northwind Labs"}]


# --- resumes, sign-in, overviews -------------------------------------------


async def test_a_resume_upload_its_read_and_a_follow_up_answer_are_each_recorded(trail):
    import httpx

    from r3.web import create_app
    from tests.test_resume import ALICE, ResumeStub, as_user
    from tests.test_resume_chat_api import GAPPY, CountingParser, _file

    db_stub = ResumeStub()
    app = create_app(database=db_stub, resume_parser=CountingParser())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, ALICE)),
                                     base_url="http://test") as c:
            rid = (await c.post("/api/resume/upload", files=_file("cv.txt", GAPPY))).json()["resume_id"]
            await c.post(f"/api/resume/{rid}/read")
            await c.post(f"/api/resume/{rid}/answer", json={"answer": "2019 - 2021"})
            refused = await c.post("/api/resume/upload", files=_file("novel.txt", "Once upon a time. " * 20))
            assert refused.status_code == 422
    rows = await _flushed(trail)
    by = {e.kind: e for e in rows}
    up = by["resume.uploaded"]
    assert (up.actor, up.user_id, up.subject) == ("user", ALICE.id, f"resume:{rid}")
    assert up.data["filename"] == "cv.txt" and up.data["chars"] > 0
    read = by["resume.read"]
    assert (read.actor, read.subject) == ("ai", f"resume:{rid}")
    assert read.data["engine"] == "heuristic" and read.data["roles"] == 2 and "faithful" in read.data
    ans = by["resume.answer"]
    assert ans.actor == "user" and ans.data["answer"] == "2019 - 2021" and ans.data["changed"]
    no = by["resume.refused"]
    assert no.data["status"] == 422 and "doesn't look like a resume" in no.summary


async def test_sign_ins_sign_ups_and_failures_are_recorded_without_passwords(trail):
    import httpx

    from r3.web import create_app
    from tests.test_auth import AuthStubDB

    app = create_app(database=AuthStubDB(), auth=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            o = {"Origin": "http://test"}
            await c.post("/signup", data={"account_name": "Morgan", "password": "pass1234pass",
                                          "password_confirm": "pass1234pass"}, headers=o)
            await c.post("/logout", headers=o)
            await c.post("/login", data={"account_name": "Morgan", "password": "wrong-one"}, headers=o)
            await c.post("/login", data={"account_name": "Morgan", "password": "pass1234pass"}, headers=o)
    rows = await _flushed(trail)
    kinds = [e.kind for e in rows if e.kind.startswith("auth.")]
    assert kinds == ["auth.signup", "auth.signout", "auth.signin_failed", "auth.signin"]
    failed = next(e for e in rows if e.kind == "auth.signin_failed")
    assert failed.user_id is None and failed.data == {"account_name": "Morgan"}
    assert all(e.user_id == 1 for e in rows if e.kind in ("auth.signup", "auth.signin"))
    assert "pass1234pass" not in repr(rows) and "wrong-one" not in repr(rows)


async def test_an_overview_records_which_engine_wrote_it(trail):
    from tests.test_job_teaser import TeaserStub
    from tests.test_job_teaser import _call as teaser_call

    db_stub = TeaserStub()
    await teaser_call(db_stub, "POST", f"/jobs/{JOB.id}/overview", "seeker", htmx=True)
    ev = next(e for e in await _flushed(trail) if e.kind == "job.overview")
    assert (ev.actor, ev.user_id, ev.subject) == ("ai", 42, f"job:{JOB.id}")
    assert ev.data["engine"] and ev.data["cached"] is False


async def test_saving_a_built_resume_is_recorded(trail):
    import httpx

    from r3.db.resume_drafts import StoredDraft
    from r3.resume.builder import Contact, Job, ResumeDraft
    from r3.web import create_app
    from tests.test_resume import ALICE, as_user
    from tests.test_resume_builder import DraftStub

    draft = ResumeDraft(
        full_name="Ann Lee", contact=Contact(email="ann@example.com", links=["github.com/ann"]),
        summary="Platform engineer who keeps systems calm. " * 3, skills=["Go", "Kubernetes"],
        experience=[Job(title="SRE", company="Acme", start="2019", end="2024",
                        bullets=["Ran on-call for 40 services across three regions."])])
    drafts = DraftStub()
    drafts.drafts[ALICE.id] = StoredDraft(user_id=ALICE.id, draft=draft.model_dump(),
                                        history=[{"role": "assistant", "text": "Hi", "topic": "name"}])
    app = create_app(database=drafts)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=as_user(app, ALICE)),
                                     base_url="http://test") as c:
            r = await c.post("/resume/build/save", data={"title": "Ann — SRE"})
            assert r.status_code == 303, r.text[:300]
    ev = next(e for e in await _flushed(trail) if e.kind == "resume.created")
    assert (ev.actor, ev.user_id) == ("user", ALICE.id) and ev.subject.startswith("resume:")
    assert ev.data["source"] == "builder" and ev.data["roles"] == 1
