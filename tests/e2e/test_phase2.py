"""Roles, resume builder and the assistant (items 7-10)."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import RESUME_TEXT, SCRIPTED_REPLY, api_paste_resume, api_signup

OPERATOR_ONLY = ["/settings", "/label"]


def test_regular_account_has_no_operator_pages(page):
    api_signup(page)  # the session's operator signed up first, so this one is regular
    page.goto("/")
    nav = page.get_by_role("navigation", name="Main")
    expect(nav.get_by_role("link", name="Jobs")).to_be_visible()
    for name in ("Health", "Label", "Sites", "Settings"):
        expect(nav.get_by_role("link", name=name, exact=True)).to_have_count(0)
    for path in OPERATOR_ONLY:
        assert page.request.get(path).status == 403, path


def test_home_checklist_changes_after_a_resume_is_added(page):
    api_signup(page)
    page.goto("/home")
    expect(page.get_by_role("heading", name="Get started")).to_be_visible()
    first = page.locator("ol.checklist > li").first
    expect(first).not_to_have_class(re.compile(r"\bdone\b"))
    expect(first.get_by_role("link", name="Upload or paste a resume")).to_be_visible()
    api_paste_resume(page, "Checklist resume", RESUME_TEXT)
    page.reload()
    expect(first).to_have_class(re.compile(r"\bdone\b"))
    expect(first.get_by_role("link", name="Checklist resume")).to_be_visible()


ANSWERS = [
    ("job title", "Site Reliability Engineer"),
    ("full name", "Jordan Example"),
    ("reach you", "jordan@example.com, Toronto, github.com/jordan-example"),
    ("most recent job", "SRE, Bluefjord Systems, Toronto, 2021, present"),
    ("What did you achieve there", "Cut paging volume by 40%\nMoved deploys to Kubernetes"),
    ("earlier job", "no"),
    ("skills and tools", "Kubernetes, Terraform, AWS, Go"),
    ("education", "BSc Computer Science, Example University, 2015"),
    ("certifications", "skip"),
    ("Anything else", "no"),
    ("sum yourself up", "An SRE who makes on-call quiet and deploys boring."),
]


def test_resume_builder_interview_reload_and_save(page):
    api_signup(page)
    page.goto("/resume/build")
    questions = page.locator("#rb-log .msg.assistant")
    count = page.locator("#rb-progress .rb-count")
    for i, (cue, answer) in enumerate(ANSWERS):
        if i == 4:  # reload mid-way: the interview continues where it was
            page.reload()
            expect(page.locator("#rb-log")).to_contain_text("Jordan Example")
        asked = questions.count()
        expect(questions.last).to_contain_text(cue)
        page.locator("#rb-input").fill(answer)
        page.locator("#rb-send").click()
        if i < len(ANSWERS) - 1:
            expect(questions).to_have_count(asked + 1)
    expect(count).not_to_contain_text(re.compile(r"^0 of"))
    page.locator("#rb-title").fill("Built with the interviewer")
    page.get_by_role("button", name="Save to my resumes").click()
    page.goto("/resume")
    row = page.locator("table.resume-list tbody tr", has_text="Built with the interviewer")
    expect(row).to_contain_text("builder")


def test_builder_echoes_the_answer_before_the_reply(page):
    api_signup(page)
    page.goto("/resume/build")
    held = []

    def hold(route):
        held.append(route)  # released below: the reply must not arrive first

    page.route(re.compile(r"/resume/build/answer"), hold)
    page.locator("#rb-input").fill("Site Reliability Engineer")
    page.locator("#rb-send").click()
    # OQ-1: the answer is echoed and the box cleared at once, while the
    # request is still held — not only after the reply lands.
    expect(page.locator("#rb-log .msg.user", has_text="Site Reliability Engineer")).to_be_visible(timeout=3_000)
    expect(page.locator("#rb-input")).to_have_value("")
    held.pop().continue_()
    expect(page.locator("#rb-log .msg.assistant").last).to_contain_text("full name")
    # OQ-4: the panel names the next step of the journey.
    expect(page.locator(".rb-save")).to_contain_text("best matches")


def _open_job(page, title: str) -> None:
    page.goto("/jobs?q=" + title.replace(" ", "+"))
    page.locator("#results td.pos a", has_text=re.compile(f"^{re.escape(title)}$")).click()
    expect(page.get_by_role("heading", name=title)).to_be_visible()


def test_assistant_streams_reply_and_transcript_does_not_leak(page):
    api_signup(page)
    page.goto("/controller")
    page.locator("#chat-input").fill("Which jobs should I look at?")
    page.locator("#chat-send").click()
    expect(page.locator("#chat-messages")).to_contain_text(SCRIPTED_REPLY)

    page.get_by_role("button", name="Sign out").click()
    api_signup(page)  # a second person signs in on the same tab
    page.goto("/controller")
    expect(page.locator("#chat-form")).to_be_visible()
    page.wait_for_load_state("networkidle")  # any transcript restore has run by now
    expect(page.locator("#chat-messages")).not_to_contain_text(SCRIPTED_REPLY)
    expect(page.locator("#chat-messages")).not_to_contain_text("Which jobs should I look at?")
