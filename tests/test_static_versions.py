"""Static files carry a version taken from their contents, so a browser
fetches a changed script after a deploy instead of reusing its cached copy.
QA (2026-10-14): after a restart, Chrome loaded the Assistant page 11 times and
never re-fetched /static/controller.js, so testers ran the old script."""

from __future__ import annotations

import hashlib
import re

from r3.web.templating import STATIC_DIR, TEMPLATES_DIR, asset_versions
from tests.test_assistant_user_ui import SEEKER, render


def test_every_static_reference_in_a_template_is_versioned():
    for path in TEMPLATES_DIR.glob("*.html"):
        bare = re.findall(r'/static/[\w.-]+\.(?:js|css)"', path.read_text())
        assert bare == [], (path.name, bare)


def test_the_version_is_a_hash_of_the_file():
    versions = asset_versions()
    expected = hashlib.sha256((STATIC_DIR / "controller.js").read_bytes()).hexdigest()[:10]
    assert versions["controller.js"] == expected


async def test_pages_link_the_versioned_files():
    html = await render(SEEKER)
    versions = asset_versions()
    for name in ("controller.js", "app.css", "app.js", "htmx.min.js", "track.js"):
        assert f'/static/{name}?v={versions[name]}"' in html, name
