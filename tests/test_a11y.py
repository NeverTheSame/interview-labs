"""Accessibility fixes (WCAG 2.2 AA audit): names for every control, focus and
announcements after htmx swaps, error wiring, skip link and landmarks. Asserts
on rendered HTML against the in-memory stub, with real sign-in cookies."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from r3.web import create_app
from r3.web.auth import SESSION_COOKIE
from tests.test_user_shell import ShellStub, _as, _resume

CSS = (Path(__file__).parents[1] / "r3/web/static/app.css").read_text()
HTML = {"accept": "text/html"}


class A11yStub(ShellStub):
    """ShellStub plus an empty resume-draft store for the build page."""

    async def ensure_resume_draft_tables(self) -> None:
        pass

    async def get_resume_draft(self, user_id):
        return None

    async def save_resume_draft(self, user_id, *, draft, history):
        pass


@pytest.fixture
def stub():
    return A11yStub()


async def _request(stub, method, path, token=None, **kw):
    app = create_app(database=stub, auth=True)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            if token:
                c.cookies.set(SESSION_COOKIE, token)
            return await c.request(method, path, **kw)


def _title(html: str) -> str:
    return re.search(r"<title>(.*?)</title>", html, re.S).group(1)


# --- A11Y-01 / 17 / 02: jobs page ----------------------------------------------


async def test_every_filter_control_has_a_label(stub):
    html = await _as(stub, "admin", "/jobs")
    controls = re.findall(r'<(?:select|input type="search")[^>]*\bid="(f-[a-z_]+)"', html)
    assert set(controls) >= {"f-q", "f-status", "f-role", "f-location", "f-employment", "f-tech", "f-site",
                             "f-days", "f-min_salary", "f-sort", "f-size"}
    for cid in controls:
        assert re.search(rf'<label class="sr-only" for="{cid}">[^<]+</label>', html), cid
    assert not re.search(r"<select(?![^>]*\bid=)", html)


async def test_jobs_page_has_a_status_region_and_pager_name(stub):
    html = await _as(stub, "seeker", "/jobs")
    assert '<div id="jobs-status" role="status" class="sr-only"></div>' in html
    assert "autofocus" not in html  # a normal page load never steals focus


async def test_row_swap_focuses_replacement_control_and_announces(stub):
    r = await _request(stub, "POST", "/jobs/7/status", "seeker", data={"status": "applied"},
                       headers={"HX-Request": "true"})
    assert r.status_code == 200
    assert re.search(r"<button[^>]*autofocus[^>]*>Undo<", r.text)
    assert 'id="jobs-status"' in r.text and 'hx-swap-oob="true"' in r.text
    assert "Marked applied: Senior ML Engineer at Acme AI" in r.text

    r = await _request(stub, "POST", "/jobs/7/status", "seeker", data={"status": "new"},
                       headers={"HX-Request": "true"})
    assert re.search(r"<button[^>]*autofocus[^>]*>Applied<", r.text)
    assert "Undone: Senior ML Engineer" in r.text


async def test_filter_swap_announces_count_and_pager_is_named(stub):
    r = await _request(stub, "GET", "/jobs", "seeker", headers={"HX-Request": "true"})
    assert 'id="jobs-status"' in r.text and 'hx-swap-oob="true"' in r.text
    assert 'aria-label="Pagination"' in (Path(__file__).parents[1] / "r3/web/templates/jobs_table.html").read_text()


async def test_row_controls_are_named_by_job_and_arrows_are_hidden(stub):
    html = await _as(stub, "seeker", "/jobs")
    row = html.split('id="job-7"')[1].split("</tr>")[0]
    suffix = '<span class="sr-only"> Senior ML Engineer at Acme AI</span>'
    for label in ("View", "Applied", "Not for me"):
        assert f"{label}{suffix}" in row, label
    admin = await _as(stub, "admin", "/jobs")
    assert '<span aria-hidden="true">↗</span><span class="sr-only"> Senior ML Engineer (opens in new tab)' in admin


# --- A11Y-03 / 08 / 10: job page -----------------------------------------------


async def test_job_page_description_region_notes_label_and_arrow(stub, monkeypatch):
    # The posting (description region, "Open posting") is on every job page
    # while R3_JOB_POSTINGS=full; with "teaser" a seeker gets the labelled
    # "About this role" teaser instead (OQ-6).
    for who in ("admin", "seeker"):
        html = await _as(stub, who, "/jobs/7")
        assert 'class="desc" tabindex="0" role="region" aria-label="Job description"' in html, who
        assert 'Open posting <span aria-hidden="true">↗</span><span class="sr-only"> (opens in new tab)</span>' in html
    monkeypatch.setenv("R3_JOB_POSTINGS", "teaser")
    html = await _as(stub, "seeker", "/jobs/7")
    assert '<section id="about" aria-labelledby="about-h">' in html and 'aria-live="polite"' in html
    assert '<h2 id="notes-h">Notes</h2>' in html
    assert re.search(r'<textarea name="notes" id="notes" aria-labelledby="notes-h"', html)


# --- A11Y-09 / 15 / 10: resume, login, signup ----------------------------------


async def test_resume_table_header_and_download_names(stub):
    stub.resumes = [_resume(3, "Dana CV", primary=True), _resume(4, "Other CV")]
    html = await _as(stub, "seeker", "/resume")
    assert '<th><span class="sr-only">Actions</span></th>' in html
    assert 'Download<span class="sr-only"> Dana CV</span>' in html
    assert 'Make primary<span class="sr-only"> Other CV</span>' in html
    assert "Error: " not in _title(html)


async def test_resume_paste_error_is_wired_to_its_field(stub):
    r = await _request(stub, "POST", "/resume/paste", "seeker", data={"title": "", "content": "short"})
    assert r.status_code == 400
    assert re.search(r'<textarea[^>]*aria-invalid="true"[^>]*aria-describedby="paste-error"[^>]*autofocus', r.text)
    assert 'id="paste-error" role="alert"' in r.text
    assert _title(r.text).startswith("Error: ")
    assert "autofocus" not in r.text.split('id="upload-file"')[1].split(">")[0]


async def test_login_error_wiring_and_title(stub):
    r = await _request(stub, "POST", "/login", data={"account_name": "", "password": ""})
    assert r.status_code == 400
    assert 'id="form-error" role="alert"' in r.text
    assert re.search(r'<input id="account_name"[^>]*aria-invalid="true"[^>]*aria-describedby="form-error"[^>]*autofocus', r.text)
    assert _title(r.text).startswith("Error: Sign in")
    r = await _request(stub, "POST", "/login", data={"account_name": "dana", "password": ""})
    assert re.search(r'<input id="password"[^>]*aria-invalid="true"[^>]*autofocus', r.text)
    assert "autofocus" not in re.search(r'<input id="account_name"[^>]*>', r.text).group(0)


async def test_signup_error_wiring_and_title(stub):
    r = await _request(stub, "POST", "/signup",
                       data={"account_name": "dana", "password": "abcd-abcd-abcd", "password_confirm": "abcd-abcd-abce"})
    assert r.status_code == 400
    assert re.search(r'<input id="password_confirm"[^>]*aria-invalid="true"[^>]*aria-describedby="form-error"[^>]*autofocus', r.text)
    assert _title(r.text).startswith("Error: Create account")
    r = await _request(stub, "POST", "/signup", data={"account_name": "x", "password": "abcd", "password_confirm": "abcd"})
    assert re.search(r'<input id="account_name"[^>]*aria-invalid="true"[^>]*aria-describedby="form-error name-hint"', r.text)
    clean = await _request(stub, "GET", "/signup")
    assert "aria-invalid" not in clean.text and not _title(clean.text).startswith("Error")
    assert re.search(r'<input id="account_name"[^>]*autofocus', clean.text)


# --- A11Y-12 / 16 / 20: layout, titles -----------------------------------------


async def test_skip_link_main_landmark_and_aria_current(stub):
    html = await _as(stub, "seeker", "/jobs")
    body = html.split("<body>")[1]
    assert body.lstrip().startswith('<a class="skip-link" href="#main">Skip to content</a>')
    assert '<main id="main" tabindex="-1">' in html
    assert re.findall(r'<a [^>]*aria-current="page"[^>]*>(\w+)</a>', html) == ["Jobs"]


async def test_error_pages_have_meaningful_titles(stub):
    r = await _request(stub, "GET", "/no/such/page", "seeker", headers=HTML)
    assert r.status_code == 404 and _title(r.text) == "Page not found · R3"
    r = await _request(stub, "GET", "/settings", "seeker", headers=HTML)
    assert r.status_code == 403 and _title(r.text) == "Operators only · R3"


# --- Controller, build page and CSS --------------------------------------------


async def test_controller_starter_tags_hidden_and_build_page_live_regions(stub):
    html = await _as(stub, "seeker", "/controller")
    assert '<span>Find</span>' not in html and '<span aria-hidden="true">Find</span>' in html
    build = await _as(stub, "seeker", "/resume/build")
    assert 'id="rb-thinking" role="status"' in build
    assert 'aria-describedby="rb-hint"' in build and 'id="rb-hint"' in build
    js = (Path(__file__).parents[1] / "r3/web/static/resume_build.js").read_text()
    assert 'getElementById("rb-log").setAttribute("aria-busy"' in js


def test_accessibility_css_is_present():
    tail = CSS.split("/* accessibility */")[1]
    for rule in (".skip-link:focus", ".desc:focus-visible", "::placeholder", ".stat.accent .l",
                 ".rb-save { grid-template-columns: minmax(0, 1fr)", ".rb-workspace *", "main p a:not(.btn)"):
        assert rule in tail, rule
