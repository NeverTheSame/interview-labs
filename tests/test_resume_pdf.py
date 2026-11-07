"""render_pdf: a plain-text resume as PDF, on both font paths (the bundled
DejaVu TTF and the latin-1 fold used when the font files are missing)."""

import re
import zlib

import pytest

from r3.resume import pdf as pdf_mod
from r3.resume.pdf import render_pdf, to_latin1

UNICODE = """Ива́н Петро́в — José Müller
ivan@example.com · “Senior” SRE ‘lead’ … 50 % uptime

ОПЫТ / EXPERIENCE
• Built Kubernetes platforms – 8 years
- Ran Prometheus → Grafana ✓
Skills:
Python, Go, Ελληνικά, 中文"""


def _page_count(data: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", data))


@pytest.mark.parametrize("unicode_font", [True, False])
def test_unicode_text_never_crashes(unicode_font):
    data = render_pdf("Иван “CV”", UNICODE, unicode_font=unicode_font)
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF")
    assert _page_count(data) == 1


def test_default_uses_bundled_font():
    assert pdf_mod._REGULAR.exists() and pdf_mod._BOLD.exists()
    data = render_pdf("CV", UNICODE)
    assert b"DejaVu" in data  # the TTF is embedded, so Cyrillic is real glyphs


def test_missing_font_files_fall_back(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_mod, "_REGULAR", tmp_path / "missing.ttf")
    data = render_pdf("CV", UNICODE)
    assert data.startswith(b"%PDF") and b"DejaVu" not in data and b"Helvetica" in data


def test_latin1_fold():
    assert to_latin1("“José” — café • Müller…") == '"José" - café - Müller...'
    assert to_latin1("Ива́н") == "????"
    assert all(ord(c) < 256 for c in to_latin1(UNICODE))


def test_layout_text_survives_on_fold_path():
    data = render_pdf("CV", "Jane Doe\n\nEXPERIENCE\n- Built things\nPlain line", unicode_font=False)
    streams = b"".join(zlib.decompress(m) for m in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S))
    for text in (b"Jane Doe", b"EXPERIENCE", b"Built things", b"Plain line"):
        assert text in streams


@pytest.mark.parametrize("unicode_font", [True, False])
def test_long_resume_paginates(unicode_font):
    body = "Jane Doe\n\n" + "\n".join(
        f"ROLE {i}\n- Delivered a long bullet about platform work number {i} " + "word " * 30 for i in range(120))
    data = render_pdf("Long", body, unicode_font=unicode_font)
    assert _page_count(data) > 5


def test_empty_content_still_renders():
    assert render_pdf("", "").startswith(b"%PDF")
