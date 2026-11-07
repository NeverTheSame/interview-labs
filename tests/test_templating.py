"""Template helpers, and that every page renders end to end with sample data."""

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import pytest

from r3.web.templating import ago, build_env, company, qs, salary, thousands, truncate

NOW = datetime.now(timezone.utc)


def test_templates_parse():
    env = build_env()
    for name in ("overview", "scout", "jobs", "job_detail", "jobs_table"):
        assert env.get_template(f"{name}.html")


@pytest.mark.parametrize("n, want", [(0, "0"), (7, "7"), (999, "999"), (1000, "1,000"), (4029, "4,029"), (56600000, "56,600,000")])
def test_thousands(n, want):
    assert thousands(n) == want


def test_salary_prefers_raw_text():
    """salary must prefer the raw posting text, because salary_min/max are
    inconsistently scaled upstream (dollars in some rows, cents in others)."""
    mn, mx = 33000000, 56600000
    assert salary(mn, mx, "USD 330,000 - 566,000 per year") == "USD 330,000 - 566,000 per year"
    assert salary(None, None, "") == "—"
    assert "from" in salary(mn, None, "")
    assert salary(mn, mx, "") == "33,000,000–56,600,000"


def test_ago_accepts_aware_and_naive():
    # A fresh clock, not the module-level NOW: the full suite runs longer
    # than a minute before this executes, and "5m ago" means 5 whole minutes.
    now = datetime.now(timezone.utc)
    assert ago(now - timedelta(minutes=5)) == "5m ago"
    assert ago(datetime.utcnow() - timedelta(hours=3)) == "3h ago"
    assert ago(now - timedelta(days=40)) == "1mo ago"
    assert ago(None) == "never"


def test_qs_preserves_filters():
    """Pagination and sort links must carry every active filter forward."""
    base = {"status": ["new"], "role": ["AI/ML"], "page": ["3"]}

    v = parse_qs(str(qs(base, page=4)).removeprefix("?"))
    assert v["page"] == ["4"]
    assert v["status"] == ["new"] and v["role"] == ["AI/ML"]
    # The original must not be mutated — it is reused for every link on a page.
    assert base["page"] == ["3"]

    # An empty value removes the key entirely.
    v2 = parse_qs(str(qs(base, status="")).removeprefix("?"))
    assert "status" not in v2


def test_truncate():
    assert truncate("short", 20) == "short"
    long = truncate("Senior Site Reliability Engineer, Platform", 20)
    assert long.endswith("…")
    assert len(long) <= 21



def test_paragraphs_keeps_plain_text_and_converts_stored_html():
    from r3.web.templating import paragraphs

    assert paragraphs("Build models.\r\n\nShip them.") == ["Build models.", "Ship them."]
    stored = ('<div class="content-intro"><h2><strong>About Acme</strong></h2><p>We build&nbsp;reliable '
              '<em>systems</em>.</p></div><h2>Must have<br></h2><ul><li>Kubernetes</li><li>Go &amp; Rust</li></ul>'
              '<script>alert(1)</script>')
    assert paragraphs(stored) == ["About Acme", "We build reliable systems.", "Must have", "Kubernetes", "Go & Rust"]
    # A stray angle bracket in prose is not markup.
    assert paragraphs("Salary < 200k, > 150k") == ["Salary < 200k, > 150k"]


@pytest.mark.parametrize("raw, shown", [
    ("Scribe (Ashby)", "Scribe"),
    ("Ironclad (alt) (Ashby)", "Ironclad"),
    ("A-place-for-mom (ashby)", "A-place-for-mom"),
    ("Acme (Greenhouse)", "Acme"), ("Acme (LEVER)", "Acme"), ("Acme (Workable) ", "Acme"),
    ("Monzo (UK)", "Monzo (UK)"), ("Pylon (YC S21)", "Pylon (YC S21)"),
    ("Altitude (Ashby) Labs", "Altitude (Ashby) Labs"), ("", ""), (None, ""),
])
def test_company_drops_only_trailing_adapter_markers(raw, shown):
    assert company(raw) == shown


def test_job_row_shows_the_company_without_adapter_markers():
    env = build_env()
    assert "{{ job.company|company|dash }}" in env.loader.get_source(env, "_job_row.html")[0]
