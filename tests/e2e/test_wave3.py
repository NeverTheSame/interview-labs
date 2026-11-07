"""Wave 3 in a real browser: job match, the accessibility structure, the 403
page and a refused run's reason."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from tests.e2e.conftest import OPERATOR, RESUME_TEXT, api_login, api_paste_resume, api_signup

# The seed's DevOps jobs list terraform, kubernetes, aws and the SRE jobs
# kubernetes, prometheus, go — all three of each in RESUME_TEXT, so both
# blocks are "3 of 3" matches. RESUME_TEXT's headline is "Site Reliability
# Engineer", so the SRE block ranks first by title fit, newest first: the
# "Site Reliability Engineer" at Northwind Labs, used below.


def test_match_sort_badges_job_line_and_home_for_a_user_with_a_resume(page):
    api_signup(page)
    api_paste_resume(page, "Match resume", RESUME_TEXT)

    page.goto("/jobs")
    assert page.locator("select[name=sort]").input_value() == "match"
    first = page.locator("#results tbody tr").first
    expect(first.locator("td.pos a")).to_have_text("Site Reliability Engineer")
    expect(first.locator(".match-badge")).to_have_text("3 of 3 skills")
    expect(page.locator(".match-hint")).to_have_count(0)

    first.locator("td.pos a").click()
    expect(page.get_by_role("heading", name="Site Reliability Engineer")).to_be_visible()
    expect(page.locator(".job-match")).to_contain_text("Your resume shows 3 of the 3 skills this posting lists.")
    expect(page.locator(".tag.matched", has_text="prometheus")).to_be_visible()

    page.goto("/home")
    expect(page.get_by_role("heading", name="Best matches for you")).to_be_visible()
    expect(page.locator("ul.best-matches li").first).to_contain_text("Site Reliability Engineer")


def test_a_user_without_a_resume_gets_the_hint_and_no_badges(page):
    api_signup(page)
    page.goto("/jobs")
    expect(page.locator(".match-hint")).to_contain_text("Add your resume")
    expect(page.locator(".match-badge")).to_have_count(0)
    assert page.locator("select[name=sort]").input_value() == "newest"
    page.goto("/home")
    expect(page.get_by_role("heading", name="Best matches for you")).to_have_count(0)


def test_the_operator_sees_no_match_ui(page):
    api_login(page, *OPERATOR)
    page.goto("/jobs")
    expect(page.locator(".match-hint")).to_have_count(0)
    expect(page.locator(".match-badge")).to_have_count(0)
    assert page.locator("select[name=sort]").input_value() != "match"
    page.goto("/")
    expect(page.get_by_role("heading", name="Best matches for you")).to_have_count(0)


def test_accessibility_structure_and_the_keyboard_status_flow(page):
    api_signup(page)
    api_paste_resume(page, "A11y resume", RESUME_TEXT)
    # "/" is the Assistant (redirects for a seeker); /home and /scout are the
    # off-nav secondary pages with no active nav item by design, so they are
    # not in this structure sweep.
    for path in ("/", "/jobs", "/resume", "/resume/build", "/controller"):
        page.goto(path)
        assert page.locator("main#main").count() == 1, path
        assert page.locator("h1").count() == 1, path
        first = page.locator("body > *").first
        assert "skip" in (first.get_attribute("class") or ""), path
        unnamed = page.evaluate("""() => [...document.querySelectorAll("input:not([type=hidden]), select, textarea")]
            .filter((c) => !(c.labels || {}).length && !c.getAttribute("aria-label") && !c.getAttribute("aria-labelledby"))
            .map((c) => c.id || c.name)""")
        assert unnamed == [], f"{path}: unnamed controls {unnamed}"
        active = page.locator(".topbar nav a[aria-current='page']")
        expect(active).to_have_count(1)
        expect(page.locator("a.skip-link")).to_be_attached()

    # Acting on a row by keyboard moves focus to Undo and announces the change.
    page.goto("/jobs")
    applied = page.locator("#results tbody tr", has_text="Site Reliability Engineer").first \
        .get_by_role("button", name=re.compile(r"^Applied"))
    applied.focus()
    applied.press("Enter")
    row = page.locator("#results tbody tr", has_text="Site Reliability Engineer").first
    expect(row.get_by_role("button", name=re.compile(r"^Undo"))).to_be_focused()
    expect(page.locator("#jobs-status")).to_contain_text("Marked applied")


def test_the_403_page_has_the_user_nav_and_a_way_home(page):
    api_signup(page)
    page.goto("/settings")
    expect(page.get_by_role("heading", name="This page is for R3 operators")).to_be_visible()
    nav = page.get_by_role("navigation", name="Main")
    expect(nav.get_by_role("link", name="Jobs")).to_be_visible()
    page.get_by_role("link", name="Back to home").click()
    expect(page).to_have_url(re.compile(r"/controller"))


def test_a_refused_run_shows_the_servers_own_reason(page):
    """A refusal in plain text (a proxy in front of the portal, say); the
    panel shows that text instead of only the status code."""
    reason = "forbidden: runs are paused on this portal"
    page.route(re.compile(r"/api/runs$"), lambda route: route.fulfill(
        status=403, content_type="text/plain", body=f"  {reason}\n")
        if route.request.method == "POST" else route.continue_())
    api_login(page, *OPERATOR)
    page.goto("/controller")
    page.get_by_role("button", name="▦ Snapshot stats").click()
    alert = page.locator("#panel-notices .notice.warn")
    expect(alert).to_contain_text("Couldn’t start the stats run")
    expect(alert.locator("p").first).to_have_text(reason)
