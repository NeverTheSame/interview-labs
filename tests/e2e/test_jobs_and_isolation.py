"""Jobs triage and per-account isolation (items 4-5)."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from tests.e2e.conftest import RESUME_TEXT, api_paste_resume, api_signup

XSS = '<img src=x onerror="alert(\'xss-img\')"><script>alert("xss-script")</script>'


def _job_link(page, title: str):
    return page.locator("#results td.pos a", has_text=re.compile(f"^{re.escape(title)}$"))


def test_list_shows_seeded_jobs_and_filters(page):
    api_signup(page)
    page.goto("/jobs")
    expect(page.locator("#results td.pos a").first).to_be_visible()
    expect(_job_link(page, "DevOps Engineer")).to_be_visible()

    page.locator("select[name=role]").select_option("SRE")
    expect(page).to_have_url(re.compile(r"role=SRE"))
    expect(_job_link(page, "Site Reliability Engineer")).to_be_visible()
    expect(_job_link(page, "DevOps Engineer")).to_have_count(0)

    page.goto("/jobs")
    search = page.get_by_placeholder(re.compile("^Search title"))
    search.fill("LLM Inference")
    search.press("Enter")
    expect(page).to_have_url(re.compile(r"q=LLM"))
    expect(page.locator("#results td.pos a")).to_have_count(1)
    expect(_job_link(page, "LLM Inference Engineer")).to_be_visible()


def _open_job(page, title: str) -> None:
    page.goto("/jobs?q=" + title.replace(" ", "+"))
    _job_link(page, title).click()
    expect(page.get_by_role("heading", name=title)).to_be_visible()


def test_status_persists_and_notes_are_inert(page):
    api_signup(page)
    _open_job(page, "Observability Engineer")
    page.get_by_role("button", name="I applied").click()
    expect(page.locator(".pill.applied").first).to_have_text("applied")
    page.reload()
    expect(page.locator(".pill.applied").first).to_have_text("applied")

    notes = page.locator("textarea[name=notes]")
    notes.fill(XSS)
    page.get_by_role("button", name="Save notes").click()
    page.reload()
    expect(page.locator("textarea[name=notes]")).to_have_value(XSS)
    expect(page.locator("img[src=x]")).to_have_count(0)
    assert page.tracked.dialogs == []


def test_one_accounts_data_is_invisible_to_another(make_page):
    a, b = make_page(), make_page()
    api_signup(a)
    api_signup(b)
    api_paste_resume(a, "A private resume", RESUME_TEXT)
    a.goto("/resume")
    a.get_by_role("link", name="A private resume", exact=True).click()
    resume_path = re.sub(r"^https?://[^/]+", "", a.url)

    _open_job(a, "Staff Site Reliability Engineer")
    a.get_by_role("button", name="I applied").click()
    expect(a.locator(".pill.applied").first).to_be_visible()
    a.locator("textarea[name=notes]").fill("A's private note")
    a.get_by_role("button", name="Save notes").click()
    job_path = re.sub(r"^https?://[^/]+", "", a.url)

    b.goto("/resume")
    expect(b.get_by_text("You have not added a resume yet.")).to_be_visible()
    assert b.request.get(resume_path).status == 404
    assert b.request.get(resume_path + "/download").status == 404
    b.goto(job_path)
    expect(b.locator(".pill.new").first).to_have_text("new")
    expect(b.locator("textarea[name=notes]")).to_have_value("")
