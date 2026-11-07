"""The claim check on the advisor's final reply (QA L16): unless the panel
started a run in this turn, a sentence saying a run started, is running or
is done, or that an error is fixed, is replaced by one honest line."""

from __future__ import annotations

import pytest
from pydantic_ai.models import override_allow_model_requests
from pydantic_ai.models.function import AgentInfo, FunctionModel

from r3.agent.contracts import ChatTurnRequest
from r3.agent.honesty import HONEST_LINE, claims_a_run, honest_reply
from r3.agent.service import Advisor
from tests.fakes import FakeRuns
from tests.fakes import test_settings as make_settings
from tests.test_routes import StubDB

NORTHWIND = "[DevOps Engineer at Northwind Labs](/jobs/1)"


@pytest.fixture(autouse=True)
def _offline():
    with override_allow_model_requests(False):
        yield


@pytest.mark.parametrize("sentence", [
    f"I'm running the stats workflow for {NORTHWIND} now.",
    "I'm running the workflow for #1 DevOps Engineer at Northwind Labs now.",
    "I've started the stats run for you.",
    "I have kicked off a run for remote SRE roles.",
    "I'll start the run now.",
    "Let me run that for you now.",
    "Your request is being processed.",
    "The stats run is now running.",
    "I fixed it, so no more 403s.",
    "The error has been fixed.",
    "Starting the workflow now.",
    f"I've applied to {NORTHWIND}.",
    "I am generating your statistics run.",
    "I went ahead and submitted the application.",
])
def test_claims(sentence):
    assert claims_a_run(sentence)


@pytest.mark.parametrize("sentence", [
    "Nothing has been created or submitted yet.",
    "No run was started.",
    "I can't start runs myself.",
    "I haven't started anything.",
    "I'm not running anything.",
    "The panel next to this chat starts the run and shows its progress there.",
    "Your stats run is still running; its results will show in the panel next to this chat.",
    "I'd start with the SRE roles at the top of [Jobs](/jobs).",
    "I'll start by explaining what each step does.",
    "R3 marked the job applied for you; nothing was sent to the employer.",
    "Start with [Build your resume](/resume/build).",
    "To start a run, ask for it here and the panel next to this chat starts it.",
    "I'm running through your resume to find the strongest points.",
    "I'll create a short list of the jobs most worth your time.",
])
def test_not_claims(sentence):
    assert not claims_a_run(sentence)


def test_a_run_looked_up_this_turn_may_be_described_as_running():
    sentence = "Your stats run is running and has counted 3 jobs so far."
    assert claims_a_run(sentence) and not claims_a_run(sentence, looked_up_runs=True)
    assert claims_a_run("I've started another run.", looked_up_runs=True)  # first person never


def test_the_claim_is_replaced_once_and_the_rest_kept():
    text = (f"Sorry about the error. I'm running the stats workflow for {NORTHWIND} now. "
            "It should finish soon, no more 403s.\n\nMeanwhile, look at [Jobs](/jobs).")
    out = honest_reply(text, run_started=False)
    assert out.startswith("Sorry about the error. " + HONEST_LINE)
    assert out.count(HONEST_LINE) == 1 and "running the stats" not in out and "403" not in out
    assert out.endswith("\n\nMeanwhile, look at [Jobs](/jobs).")


def test_a_list_item_keeps_its_marker():
    out = honest_reply("Next:\n- I've started the stats run for you.\n- Check [Jobs](/jobs).", run_started=False)
    assert out == f"Next:\n- {HONEST_LINE}\n- Check [Jobs](/jobs)."


def test_honest_text_and_a_run_started_this_turn_are_left_alone():
    plain = "You have 3 new jobs. Start with [Senior SRE at Acme](/jobs/7)."
    assert honest_reply(plain, run_started=False) == plain
    started = "I've started a stats run; its progress shows in the panel next to this chat."
    assert honest_reply(started, run_started=True) == started


def test_a_reply_that_is_only_a_claim_becomes_the_honest_line():
    assert honest_reply("I'm running it now.", run_started=False) == HONEST_LINE


def _model(text: str) -> FunctionModel:
    async def stream(messages, info: AgentInfo):
        yield text

    return FunctionModel(stream_function=stream)


async def _turn(message: str, reply: str, *, user=None, **settings) -> list:
    advisor = Advisor(make_settings(router_mode="rules", **settings), models={"chat": _model(reply)})
    turn = ChatTurnRequest(messages=[{"role": "user", "content": message}])
    return [e async for e in advisor.stream_turn(turn, StubDB(), FakeRuns(), user=user)]


async def test_a_false_claim_never_reaches_the_final_reply():
    claim = "I'm running the workflow for #1 DevOps Engineer at Northwind Labs now."
    events = await _turn("it failed with a 403, why?", f"Sorry. {claim}")
    final = next(e for e in events if e.type == "final")
    assert final.text == f"Sorry. {HONEST_LINE}"


async def test_a_run_started_this_turn_may_be_narrated():
    reply = "I've started a stats run; progress shows in the panel next to this chat."
    events = await _turn("show me the stats", reply)
    action = next(e for e in events if e.type == "action").action
    assert action.kind == "stats" and action.run is not None
    assert next(e for e in events if e.type == "final").text == reply


# --- application outcomes (QA D2, 2026-10-14) -------------------------------------------
# R3 only records that the user applied; it never sends anything. The advisor still answered "did my application go through?" with "Yes, your application went
# through!", so a claim that an application reached the employer is replaced too.

from r3.agent.honesty import APPLY_LINE  # noqa: E402


@pytest.mark.parametrize("claim", [
    "Yes, your application went through!",
    "Your application to Northwind Labs has been submitted.",
    "I submitted your application for the DevOps role.",
    "Your resume was sent to the employer.",
    "Northwind Labs has received your application.",
    "You've successfully applied to all three jobs!",
])
def test_a_claim_that_an_application_reached_the_employer_is_replaced(claim):
    out = honest_reply(f"{claim} Status: Applied (completed).", run_started=True)
    assert claim not in out and APPLY_LINE in out and "Status: Applied (completed)." in out


@pytest.mark.parametrize("truth", [
    "Nothing was sent to the employer.",
    "R3 doesn't submit applications; it only marks the job applied.",
    "Your application has not been submitted anywhere.",
    "It didn't go through to the employer: R3 only records it.",
    "You marked it applied on Tuesday.",
    "Once your application is submitted on their site, follow up in a week.",
    "If your resume was sent through a referral, mention it.",
    "Your application wasn't submitted; R3 only recorded it.",
])
def test_true_statements_about_applications_stay(truth):
    assert honest_reply(truth, run_started=False) == truth


@pytest.mark.parametrize("who", ["DANA", "ADMIN"])
async def test_nobody_is_told_their_application_went_through(who):
    from tests import test_agent_users

    reply = "Yes, your application went through! Status: Applied (completed)."
    user = getattr(test_agent_users, who)
    final = next(e for e in await _turn("did my application go through?", reply, user=user) if e.type == "final")
    assert final.text == f"{APPLY_LINE} Status: Applied (completed)."
