"""The Assistant's replies link R3's own pages (OQ-7, QA M8/L4): a Markdown
link to a same-site path ("[Build your resume](/resume/build)", a job as
"[Position at Company](/jobs/1)") becomes an anchor under the page's base
path that opens in the same tab. A javascript:, data:, protocol-relative or
backslash target stays plain text. The chat stream is answered by the test
(page.route), so what is under test is controller.js's renderer."""

from __future__ import annotations

import re

from playwright.sync_api import expect

from r3.agent.contracts import AdviceAction, RouteDecision
from r3.agent.events import ActionEvent, DeltaEvent, DoneEvent, FinalEvent, RouteEvent, sse
from tests.e2e.conftest import api_signup

SAFE = "Start with [Build your resume](/resume/build), then [DevOps Engineer at Northwind Labs](/jobs/1)."
UNSAFE = ("Not links: [evil](javascript:alert(1)), [proto](//evil.example/x), [slash](/\\evil.example/x), "
          "[data](data:text/html,hi).")
REPLY = f"{SAFE}\n\n{UNSAFE}"


def _stream(route) -> None:
    body = "".join(sse(e) for e in (
        RouteEvent(decision=RouteDecision(intent="advice", router="rules", reason="scripted by the test")),
        ActionEvent(action=AdviceAction()), DeltaEvent(text=REPLY),
        FinalEvent(text=REPLY, model="openai:gpt-4o-mini"), DoneEvent()))
    route.fulfill(status=200, headers={"Content-Type": "text/event-stream"}, body=body)


def _ask(page) -> object:
    page.locator("#chat-input").fill("Help me build my resume")
    page.locator("#chat-send").click()
    reply = page.locator("#chat-messages article.msg.assistant .msg-body").last
    expect(reply).to_contain_text("Not links:")
    return reply


def _hrefs(reply) -> list[str]:
    return reply.locator("a").evaluate_all("els => els.map(e => e.getAttribute('href'))")


def test_reply_links_open_portal_pages_and_nothing_else(page):
    api_signup(page)
    page.route(re.compile(r"/api/chat/stream$"), _stream)
    page.goto("/controller")
    reply = _ask(page)
    assert _hrefs(reply) == ["/resume/build", "/jobs/1"]
    builder = reply.get_by_role("link", name="Build your resume")
    expect(builder).not_to_have_attribute("target", "_blank")
    expect(reply.get_by_role("link", name="DevOps Engineer at Northwind Labs")).to_be_visible()
    for text in ("evil", "proto", "slash", "data"):
        expect(reply).to_contain_text(text)
    expect(reply).not_to_contain_text("#1")
    builder.click()
    expect(page).to_have_url(re.compile(r"/resume/build$"))


def test_reply_links_keep_the_base_path(page):
    """Under a path prefix (R3_ROOT_PATH, behind a reverse proxy) a
    page link carries the prefix the page was rendered with."""
    api_signup(page)

    def prefixed(route) -> None:
        response = route.fetch()
        route.fulfill(response=response, body=response.text().replace('data-base-path=""', 'data-base-path="/r3pfx"'))

    page.route(re.compile(r"/controller$"), prefixed)
    page.route(re.compile(r"/r3pfx/api/chat/stream$"), _stream)
    page.route(re.compile(r"/r3pfx/(?!api/chat/stream$)"),
               lambda route: route.continue_(url=route.request.url.replace("/r3pfx/", "/", 1)))
    page.goto("/controller")
    assert page.locator("#r3-controller").get_attribute("data-base-path") == "/r3pfx"
    assert _hrefs(_ask(page)) == ["/r3pfx/resume/build", "/r3pfx/jobs/1"]
