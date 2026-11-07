"""The job teaser in the browser (OQ-6): a regular account sees that a job
exists, whether it is live, its stack and an overview on request, never the
posting, a link to it or its source board. The overview is cached in
r3_job_summaries on real Postgres. The operator keeps the posting."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from tests.e2e.conftest import OPERATOR, OVERVIEW_TEXT, api_login, api_signup
from tests.e2e.test_jobs_and_isolation import _open_job

OVERVIEW_BUTTON = "Show an overview of this role"


def test_a_seeker_gets_the_overview_and_never_the_posting(page, teaser_mode):
    api_signup(page)
    page.goto("/jobs")
    expect(page.locator("table.jobs th", has_text=re.compile("^(Source|Site)$"))).to_have_count(0)
    expect(page.locator("#results")).not_to_contain_text("northwind-labs")

    _open_job(page, "Technical Writer")
    expect(page.get_by_role("link", name=re.compile("Open posting"))).to_have_count(0)
    expect(page.locator("main a[href^='http']")).to_have_count(0)
    expect(page.locator("main")).not_to_contain_text("synthetic job posting written for")
    expect(page.locator("main")).not_to_contain_text("northwind-labs.example")
    expect(page.locator("dl.meta")).to_contain_text("Live")
    expect(page.locator("main .tags")).to_contain_text("python")

    page.get_by_role("button", name=OVERVIEW_BUTTON).click()
    expect(page.locator("#job-overview")).to_have_text(OVERVIEW_TEXT)
    expect(page.locator("#job-overview p")).to_be_focused()

    page.reload()  # served from the cache now
    expect(page.locator("#job-overview")).to_have_text(OVERVIEW_TEXT)
    expect(page.get_by_role("button", name=OVERVIEW_BUTTON)).to_have_count(0)
    assert page.tracked.console_errors == [] and page.tracked.page_errors == []


def test_an_expired_job_says_it_is_no_longer_live(page):
    api_signup(page)
    _open_job(page, "Support Engineer")
    expect(page.locator("dl.meta")).to_contain_text("No longer live")


def test_the_operator_keeps_the_posting(page):
    api_login(page, *OPERATOR)
    _open_job(page, "Technical Writer")
    expect(page.get_by_role("link", name=re.compile("Open posting"))).to_be_visible()
    expect(page.get_by_role("heading", name="Description")).to_be_visible()
    expect(page.locator("main")).to_contain_text("synthetic job posting written for")
    expect(page.get_by_role("heading", name="About this role")).to_have_count(0)
