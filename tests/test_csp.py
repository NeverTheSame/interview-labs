"""Content-Security-Policy and friends on every response (R3-03 / X-02)."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from r3.web.headers import CSP
from tests.test_security import _app, _client

TEMPLATES = Path(__file__).parent.parent / "r3" / "web" / "templates"


@pytest.fixture
async def client():
    async for c in _client(_app()):
        yield c


@pytest.mark.parametrize("path", ["/", "/jobs", "/jobs/7", "/sites", "/controller", "/healthz", "/static/app.js", "/nope"])
async def test_every_response_carries_the_policy(client, path):
    r = await client.get(path)
    assert r.headers["content-security-policy"] == CSP
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"


def test_policy_blocks_inline_script_and_embedding():
    directives = {d.split()[0]: d.split()[1:] for d in CSP.split("; ")}
    assert directives["script-src"] == ["'self'"]
    assert directives["object-src"] == ["'none'"]
    assert directives["base-uri"] == ["'self'"] and directives["form-action"] == ["'self'"]
    assert directives["frame-ancestors"] == ["'self'"]


def test_templates_have_no_inline_script_or_handlers():
    """What the policy forbids must not be in the templates, or the page breaks."""
    bad = []
    for p in TEMPLATES.glob("*.html"):
        text = p.read_text()
        for m in re.finditer(r"<script\b([^>]*)>", text):
            attrs = m.group(1)
            if "src=" not in attrs and "application/json" not in attrs:
                bad.append(f"{p.name}: inline <script>")
        if re.search(r"\son[a-z]+\s*=\s*[\"']", text):
            bad.append(f"{p.name}: inline event handler")
        if re.search(r"javascript:", text, re.I):
            bad.append(f"{p.name}: javascript: URL")
    assert not bad, bad


async def test_job_page_renders_without_inline_script(client):
    r = await client.get("/jobs/7")
    assert r.status_code == 200
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", r.text)
