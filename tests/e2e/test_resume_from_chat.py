"""A resume made from what the person typed in the Assistant chat, in the
browser: the turn hands the text to the page, which saves it, shows what R3
read and asks for what is missing, like an upload; a refusal says what to do,
with working links."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from r3.agent.contracts import ResumeFromTextAction, RouteDecision
from r3.agent.events import ActionEvent, DoneEvent, FinalEvent, RouteEvent
from r3.agent.router import RESUME_HELP
from r3.agent.service import MAKING_RESUME
from tests.e2e.conftest import api_signup
from tests.e2e.test_resume_upload import GAPPY, _answer


def _turn(text: str) -> list[object]:
    return [RouteEvent(decision=RouteDecision(intent="advice", router="rules", reason=RESUME_HELP)),
            ActionEvent(action=ResumeFromTextAction(text=text)), FinalEvent(text=MAKING_RESUME), DoneEvent()]


def test_a_background_typed_in_the_chat_becomes_a_resume_with_follow_ups(page, scripted_events):
    scripted_events(_turn(GAPPY))
    api_signup(page)
    page.goto("/controller")
    _answer(page, "make me a resume from this:\n" + GAPPY)

    log = page.locator("#chat-messages")
    expect(log).to_contain_text(MAKING_RESUME)
    card = page.locator(".resume-card")
    expect(card).to_contain_text("Here’s what I read")
    expect(card).to_contain_text("Platform Engineer — Globex · 2020 – Present")
    expect(log).to_contain_text("It is now your primary resume")
    expect(log).to_contain_text("When did you start and finish as Support Engineer at Hooli?")
    expect(page.locator("#chat-input")).to_have_attribute("placeholder", re.compile("Answer R3"))

    _answer(page, "2018 - 2020")
    expect(log).to_contain_text("LinkedIn, GitHub or a portfolio")

    page.goto("/resume")
    expect(page.locator("table")).to_contain_text("chat")


def test_a_background_r3_cannot_read_says_what_to_do_with_links(page, scripted_events):
    prose = ("I have been a backend developer for six years, mostly on payment systems and internal tools, "
             "and I want a remote role where I can keep doing that kind of work with a small team.")
    scripted_events(_turn(prose))
    api_signup(page)
    page.goto("/controller")
    _answer(page, "make me a resume from this: " + prose)

    alert = page.locator("#chat-messages .msg-error").last
    expect(alert).to_contain_text("needs the AI model")
    expect(alert.get_by_role("link", name="Build your resume")).to_have_attribute("href", re.compile(r"/resume/build$"))
    page.goto("/resume")
    expect(page.locator(".resume-empty")).to_be_visible()  # nothing was saved
