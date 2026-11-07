"""The Jinja environment and the helpers available inside every template."""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from markupsafe import Markup

from r3.text import plain_description, safe_http_url

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"


def asset_versions(static_dir: Path = STATIC_DIR) -> dict[str, str]:
    """A short hash of each static file's contents. Templates link
    ``/static/<name>?v=<hash>`` (the ``asset`` global), so a deploy that
    changes a file changes its URL and browsers fetch it instead of reusing a
    cached copy. Read once when the templates load; a restart re-reads them."""
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:10]
            for f in static_dir.iterdir() if f.is_file()}


def _as_time(v: object) -> datetime | date | None:
    """Normalize the nullable / non-nullable timestamp mix the row models
    carry, so templates can call the same helper on either. Aware datetimes
    are shown in UTC."""
    if isinstance(v, datetime):
        return v.astimezone(timezone.utc) if v.tzinfo else v
    if isinstance(v, date):
        return v
    return None


def fmt_date(v: object) -> str:
    """Render a nullable timestamp as a short date, or an em dash."""
    t = _as_time(v)
    return t.strftime("%Y-%m-%d") if t else "—"


def fmt_datetime(v: object) -> str:
    t = _as_time(v)
    return t.strftime("%Y-%m-%d %H:%M") if t else "—"


def ago(v: object) -> str:
    """Render how long ago a timestamp was, in the coarsest useful unit."""
    t = _as_time(v)
    if t is None:
        return "never"
    if not isinstance(t, datetime):
        t = datetime(t.year, t.month, t.day)
    now = datetime.now(timezone.utc) if t.tzinfo else datetime.utcnow()
    secs = (now - t).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    if secs < 30 * 86400:
        return f"{int(secs // 86400)}d ago"
    return f"{int(secs // (30 * 86400))}mo ago"


def dash(s: str | None) -> str:
    """Print an em dash for empty strings so table cells never look broken."""
    return s if s and s.strip() else "—"


def truncate(s: str, n: int) -> str:
    """Shorten a string for table display, on a word boundary."""
    if len(s) <= n:
        return s
    cut = s[:n]
    i = cut.rfind(" ")
    if i > n // 2:
        cut = cut[:i]
    return cut + "…"


def host(raw: str) -> str:
    """Extract the display hostname from a URL."""
    netloc = urlsplit(raw).netloc
    if not netloc:
        return raw
    return netloc.removeprefix("www.")


def thousands(n: int) -> str:
    """Format an integer with comma separators."""
    return f"{n:,}"


def salary(mn: int | None, mx: int | None, raw: str) -> str:
    """Prefer the raw string scraped from the posting, because the parsed
    salary_min/salary_max columns are inconsistently scaled upstream — some
    rows store dollars, others cents (e.g. 56600000 for $566,000) — so
    rendering them would misstate roughly one row in six. The parsed columns
    are still used for filtering and sorting, where approximate magnitude is
    good enough."""
    if raw and raw.strip():
        return raw.strip()
    if mn is not None and mx is not None:
        return f"{thousands(mn)}–{thousands(mx)}"
    if mx is not None:
        return "up to " + thousands(mx)
    if mn is not None:
        return "from " + thousands(mn)
    return "—"


def paragraphs(s: str) -> list[str]:
    """Split a description into paragraphs for the detail and labeling
    views. A description stored as HTML is turned into text first
    (:func:`r3.text.plain_description`); it is never rendered as
    markup."""
    return [p.strip() for p in plain_description(s).replace("\r\n", "\n").split("\n") if p.strip()]


def qs(base: dict[str, list[str]], **overrides: object) -> Markup:
    """Rebuild the current query string with some keys overridden. Used by
    pagination and sort links so they preserve every active filter. An empty
    override removes the key."""
    nxt = {k: list(v) for k, v in base.items()}
    for k, v in overrides.items():
        if v is None or v == "":
            nxt.pop(k, None)
        else:
            nxt[k] = [str(v)]
    return Markup("?" + urlencode(nxt, doseq=True))


# Trailing "(Ashby)", "(alt)" and the like that scraper adapters append to a
# company name; a real "(UK)" or "(YC S21)" is not one of them.
_ADAPTER_SUFFIX = re.compile(r"(\s*\((?:ashby|greenhouse|lever|workable|alt)\))+\s*$", re.IGNORECASE)


def company(name: str) -> str:
    """A company name without the scraper's adapter markers."""
    return _ADAPTER_SUFFIX.sub("", name or "")


def pct(part: int, whole: int) -> str:
    if whole == 0:
        return "0%"
    return f"{part / whole * 100:.0f}%"


def build_env(base_path: str = "") -> Environment:
    """Build the template environment. Templates are loaded up front so a
    broken one fails at startup rather than on a request."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        date=fmt_date,
        datetime=fmt_datetime,
        ago=ago,
        dash=dash,
        truncate=truncate,
        host=host,
        thousands=thousands,
        paragraphs=paragraphs,
        company=company,
        safe_url=safe_http_url,
    )
    versions = asset_versions()
    env.globals.update(qs=qs, salary=salary, pct=pct, host=host, base_path=base_path,
                       asset=lambda name: f"{name}?v={versions[name]}" if name in versions else name)
    for name in env.list_templates():
        env.get_template(name)
    return env
