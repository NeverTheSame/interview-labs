"""Every page a signed-in account can open renders cleanly at desktop and
phone widths: no server error, no console error, no horizontal overflow (item 6)."""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import OPERATOR, RESUME_TEXT, api_login, api_paste_resume, api_signup

VIEWPORTS = {"desktop": {"width": 1280, "height": 900}, "phone": {"width": 390, "height": 844}}
SEEKER_PAGES = ["/", "/controller", "/jobs", "/scout", "/resume", "/resume/build", "{job}", "{resume}"]
OPERATOR_PAGES = SEEKER_PAGES + ["/label", "/settings"]

# Open layout bugs, asserted to STILL overflow so the fix is noticed (the test
# fails then, and the entry is deleted). Every other page must not overflow.
KNOWN_OVERFLOW: dict = {}


def _first_href(page, path: str, selector: str) -> str:
    page.goto(path)
    return page.locator(selector).first.get_attribute("href")


def _check_pages(page, paths: list[str]) -> None:
    job = _first_href(page, "/jobs", "#results td.pos a")
    resume = _first_href(page, "/resume", "table.resume-list td a.strong")
    problems = []
    for p in paths:
        url = p.format(job=job, resume=resume)
        page.tracked.console_errors.clear()
        page.tracked.page_errors.clear()
        resp = page.goto(url)
        page.wait_for_load_state("networkidle")
        overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        if resp is None or resp.status >= 400:
            problems.append(f"{url}: HTTP {resp.status if resp else 'none'}")
        if page.tracked.console_errors or page.tracked.page_errors:
            problems.append(f"{url}: console {page.tracked.console_errors} page {page.tracked.page_errors}")
        known = KNOWN_OVERFLOW.get((p, page.viewport_size["width"]))
        if known and overflow <= 0:
            problems.append(f"{url}: no longer overflows; remove KNOWN_OVERFLOW entry ({known})")
        elif overflow > 0 and not known:
            problems.append(f"{url}: document overflows horizontally by {overflow}px")
        expect(page.locator("body")).not_to_contain_text("Internal Server Error")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("viewport", VIEWPORTS)
def test_seeker_pages_render_cleanly(make_page, viewport):
    page = make_page(VIEWPORTS[viewport])
    api_signup(page)
    api_paste_resume(page, "Render check", RESUME_TEXT)
    _check_pages(page, SEEKER_PAGES)


@pytest.mark.parametrize("viewport", VIEWPORTS)
def test_operator_pages_render_cleanly(make_page, viewport):
    page = make_page(VIEWPORTS[viewport])
    api_login(page, *OPERATOR)
    if page.request.get("/resume").text().count("resume-list") == 0:
        api_paste_resume(page, "Operator render check", RESUME_TEXT)
    _check_pages(page, OPERATOR_PAGES)
    expect(page.get_by_role("navigation", name="Main").get_by_role("link", name=re.compile("^Settings$"))).to_be_visible()
