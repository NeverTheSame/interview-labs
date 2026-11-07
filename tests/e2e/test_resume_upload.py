"""Resume upload in the Assistant chat, in the browser: the Upload resume
button and drag and drop stage a file as a chip, Upload saves it, R3 shows
what it read and asks for the missing role dates and profile link right in
the chat; Skip leaves, and the Resume page brings the seeker back."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from tests.e2e.conftest import SCRIPTED_REPLY, api_signup

GAPPY = ("Sam Lee\nsam@example.com\n\nWork History\nPlatform Engineer | Globex | 2020 - Present\n"
         "- Ran the platform for three product teams.\nSupport Engineer at Hooli\n- Answered tickets every day.\n\n"
         "Skills\nGo; Rust; Kubernetes")


def _pick(page, name="sam-cv.txt", text=GAPPY):
    page.set_input_files("#resume-file", files=[{"name": name, "mimeType": "text/plain", "buffer": text.encode()}])


def _answer(page, text):
    page.locator("#chat-input").fill(text)
    page.locator("#chat-input").press("Enter")


def _drop(page, selector, names):
    """Dispatch dragenter, dragover and drop carrying files, as a real drag would."""
    page.evaluate("""([selector, names, text]) => {
        const dt = new DataTransfer();
        for (const n of names) dt.items.add(new File([text], n, { type: "text/plain" }));
        const target = document.querySelector(selector);
        for (const type of ["dragenter", "dragover", "drop"]) {
            target.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
        }
    }""", [selector, names, GAPPY])


def test_the_upload_button_reads_the_resume_and_fills_the_gaps_in_the_chat(page):
    api_signup(page)
    page.goto("/controller")
    expect(page.get_by_role("button", name="Upload resume")).to_be_visible()
    _pick(page)
    chip = page.locator("#resume-chip")
    expect(chip).to_contain_text("sam-cv.txt")
    page.get_by_role("button", name="Upload sam-cv.txt").click()

    card = page.locator(".resume-card")
    expect(card).to_contain_text("Here’s what I read")
    expect(card).to_contain_text("Sam Lee")
    expect(card).to_contain_text("Platform Engineer — Globex · 2020 – Present")
    expect(card.locator(".resume-checks")).to_contain_text("missing: Profile link")
    expect(chip).to_be_hidden()
    expect(page.locator("#chat-messages")).to_contain_text("When did you start and finish as Support Engineer at Hooli?")
    expect(page.locator("#chat-input")).to_have_attribute("placeholder", re.compile("Answer R3"))

    _answer(page, "whenever")
    expect(page.locator("#chat-messages")).to_contain_text("I couldn't find a start and end date in that")
    _answer(page, "what do you need?")  # a question back goes to the assistant, not the canned reply
    expect(page.locator("#chat-messages")).to_contain_text(SCRIPTED_REPLY)
    expect(page.locator("#chat-input")).to_have_attribute("placeholder", re.compile("Answer R3"))
    _answer(page, "2017 - present")
    expect(page.locator("#chat-messages")).to_contain_text("Got it: Support Engineer at Hooli: 2017 – present.")
    expect(page.locator("#chat-messages")).to_contain_text("LinkedIn, GitHub or a portfolio")
    page.reload()  # the follow-up survives a reload
    expect(page.locator("#chat-input")).to_have_attribute("placeholder", re.compile("Answer R3"))
    _answer(page, "github.com/samlee")
    expect(page.locator("#chat-messages")).to_contain_text("Your resume is complete")
    expect(page.get_by_role("link", name="Open Jobs")).to_be_visible()

    page.goto("/resume")
    page.get_by_role("link", name="sam-cv", exact=True).click()
    expect(page.locator("main")).not_to_contain_text("Missing:")
    expect(page.locator("textarea[name=content]")).to_have_value(re.compile("github.com/samlee"))
    assert page.tracked.console_errors == [] and page.tracked.page_errors == []


def test_drag_and_drop_stages_one_file_and_never_navigates_away(page):
    api_signup(page)
    page.goto("/controller")
    url = page.url
    _drop(page, "#r3-chat", ["cv.txt"])
    expect(page.locator("#resume-chip")).to_contain_text("cv.txt")
    page.get_by_role("button", name="Remove cv.txt").click()
    expect(page.locator("#resume-chip")).to_be_hidden()

    _drop(page, "#r3-chat", ["a.txt", "b.txt"])
    expect(page.locator("#chat-error")).to_have_text("Drop one resume at a time.")
    _drop(page, "#r3-chat", ["virus.exe"])
    expect(page.locator("#chat-error")).to_contain_text("not a file R3 can read")
    _drop(page, ".topbar", ["cv.txt"])
    expect(page.locator("#chat-error")).to_have_text("Drop the file on the chat to upload it.")
    assert page.url == url and page.locator("#resume-chip").is_hidden()


def test_a_refused_file_keeps_the_chip_and_skip_comes_back_from_the_resume_page(page):
    api_signup(page)
    page.goto("/controller")
    _pick(page, "notes.txt", "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 20)
    page.get_by_role("button", name="Upload notes.txt").click()
    expect(page.locator("#chat-messages .msg-error")).to_contain_text("doesn't look like a resume")
    page.get_by_role("button", name="Remove notes.txt").click()

    _pick(page)
    page.get_by_role("button", name="Upload sam-cv.txt").click()
    page.get_by_role("button", name="Skip for now").click()
    expect(page.locator("#chat-messages")).to_contain_text("you can finish it any time")
    page.get_by_role("link", name="Open the resume").last.click()
    expect(page.locator(".resume-missing")).to_contain_text("dates for Support Engineer; a profile link")
    page.get_by_role("link", name="Finish in the Assistant").click()
    expect(page.locator("#chat-messages")).to_contain_text("When did you start and finish as Support Engineer at Hooli?")
