"""Signed-out visitors, sign up, sign out, sign in (coverage items 1-2)."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlsplit

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import PASSWORD, api_signup, unique_name


@pytest.mark.parametrize("path", ["/", "/jobs", "/resume", "/controller", "/jobs/1"])
def test_signed_out_visitor_is_sent_to_login_with_next(page, path):
    page.goto(path)
    expect(page.get_by_role("heading", name="Sign in to R3")).to_be_visible()
    # "/" is where sign-in lands anyway, so the app may drop next=/ from the URL.
    expected = "" if path == "/" else path
    expect(page.locator("input[name=next]")).to_have_value(re.compile(f"^({re.escape(expected)}|/)$"
                                                                      if path == "/" else f"^{re.escape(path)}$"))
    assert urlsplit(page.url).path == "/login"
    assert parse_qs(urlsplit(page.url).query).get("next", [expected])[0] in (expected, "/")


def test_login_page_explains_the_product(page):
    page.goto("/login")
    # The page says what R3 is before asking for credentials.
    assert re.search(r"job", page.locator("main, body").first.inner_text(), re.I)
    expect(page.get_by_role("link", name=re.compile("create (an )?account", re.I)).first).to_be_visible()


def test_healthz_is_open(page):
    r = page.request.get("/healthz")
    assert r.status == 200


def _signup_form(page, name: str, password: str, confirm: str) -> None:
    page.goto("/signup")
    page.get_by_label("Account name").fill(name)
    page.get_by_label("Password", exact=True).fill(password)
    page.get_by_label("Confirm password").fill(confirm)
    page.get_by_role("button", name="Create account").click()


def test_signup_validation_then_success(make_page):
    taken = api_signup(make_page())  # another visitor already owns this name
    page = make_page()
    alert = page.get_by_role("alert")

    _signup_form(page, unique_name(), PASSWORD, PASSWORD + "x")
    expect(alert).to_have_text("The two passwords do not match.")
    _signup_form(page, "ab", PASSWORD, PASSWORD)
    expect(alert).to_contain_text("Account name must be 3–40 characters")
    _signup_form(page, taken.upper(), PASSWORD, PASSWORD)
    expect(alert).to_have_text("That account name is already taken.")
    _signup_form(page, unique_name(), "abc", "abc")
    expect(alert).to_contain_text("Password must be at least 12 characters")

    name = unique_name()
    _signup_form(page, name, PASSWORD, PASSWORD)
    expect(page).not_to_have_url(re.compile(r"/(signup|login)"))
    expect(page.locator(".account-name")).to_have_text(name)


def test_sign_out_sign_in_and_next_handling(page):
    name = api_signup(page)
    page.goto("/jobs")
    page.get_by_role("button", name="Sign out").click()
    expect(page).to_have_url(re.compile(r"/login"))

    # The session really ended: history and direct URLs both land on login.
    # Pages are no-store, so Back refetches /jobs instead of showing a cached copy.
    page.go_back()
    expect(page).to_have_url(re.compile(r"/login"))
    expect(page.locator(".account-name")).to_have_count(0)
    page.goto("/resume")
    expect(page).to_have_url(re.compile(r"/login\?next="))

    page.get_by_label("Account name").fill(name)
    page.get_by_label("Password").fill("wrong-password")
    page.get_by_role("button", name="Sign in").click()
    expect(page.get_by_role("alert")).to_have_text("Account name or password is incorrect.")

    page.goto("/login?next=/resume")
    page.get_by_label("Account name").fill(name)
    page.get_by_label("Password").fill(PASSWORD)
    page.get_by_role("button", name="Sign in").click()
    expect(page).to_have_url(re.compile(r"/resume$"))
    expect(page.locator(".account-name")).to_have_text(name)

    page.get_by_role("button", name="Sign out").click()
    page.goto("/login?next=//evil.example/steal")
    page.get_by_label("Account name").fill(name)
    page.get_by_label("Password").fill(PASSWORD)
    page.get_by_role("button", name="Sign in").click()
    expect(page.locator(".account-name")).to_have_text(name)
    assert page.url.startswith(page.context.request.get("/healthz").url.rsplit("/healthz", 1)[0])
    assert "evil.example" not in page.url
