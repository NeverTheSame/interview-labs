"""Stored URLs never become a script: the shared helper, the Jinja filter
and the rendered pages (R3-03)."""

from __future__ import annotations

import pytest

from r3.text import safe_http_url
from r3.web.templating import build_env

BAD = [
    "javascript:alert(1)", "JavaScript:alert(1)", " \tjavascript:alert(1)", "java\tscript:alert(1)",
    "java\nscript:alert(1)", "\x01javascript:alert(1)", "data:text/html,<script>1</script>", "vbscript:x",
    "file:///etc/passwd", "//evil.example/x", "ftp://x.example/", "", "   ", "http://", "https:///path",
]


@pytest.mark.parametrize("url", BAD)
def test_safe_http_url_rejects(url):
    assert safe_http_url(url) == ""


@pytest.mark.parametrize("url", ["https://a.example/x?y=1", "HTTP://A.example/", "  https://a.example/j \n"])
def test_safe_http_url_keeps_web_links(url):
    out = safe_http_url(url)
    assert out.lower().startswith(("http://", "https://")) and out == out.strip()


def test_safe_url_filter_in_templates():
    env = build_env()
    t = env.from_string("<a href=\"{{ u|safe_url }}\">x</a>")
    assert t.render(u="javascript:alert(1)") == '<a href="">x</a>'
    assert t.render(u="https://a.example/?a=1&b=2") == '<a href="https://a.example/?a=1&amp;b=2">x</a>'
    assert t.render(u=None) == '<a href="">x</a>'


def test_no_template_prints_a_stored_url_without_the_filter():
    import re
    from pathlib import Path
    root = Path(__file__).parent.parent / "r3" / "web" / "templates"
    bad = []
    for p in root.glob("*.html"):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r'(?:href|src)="\{\{([^}]*)\}\}', line):
                expr = m.group(1)
                if re.search(r"job_url|apply_url|site_url|\bs\.url|ui_url", expr) and "safe_url" not in expr \
                        and "urlencode" not in expr:
                    bad.append(f"{p.name}:{n}")
    assert not bad, bad


def test_no_template_needs_javascript_eval():
    """The Content-Security-Policy's script-src 'self' forbids eval, and htmx
    evaluates trigger filters (``keyup[...]``, ``every 4s [...]``) and
    ``js:`` values with it, so none may appear: they would throw on every page
    load and silently stop working."""
    import re
    from pathlib import Path
    root = Path(__file__).parent.parent / "r3" / "web" / "templates"
    bad = []
    for p in root.glob("*.html"):
        for n, line in enumerate(p.read_text().splitlines(), 1):
            for m in re.finditer(r'hx-trigger="([^"]*)"', line):
                if "[" in m.group(1):
                    bad.append(f"{p.name}:{n}")
            if re.search(r'hx-(vals|vars|headers)="\s*(js|javascript):', line):
                bad.append(f"{p.name}:{n}")
    assert not bad, bad
