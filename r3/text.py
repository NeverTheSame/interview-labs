"""Small text helpers the pages, the job overview and the resume reader
share: a stored posting description as readable text, the head of a long
one, the one gate for links rendered into an ``href``, and the check that a
model's rewrite of a resume added no facts."""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import urlsplit

from selectolax.parser import HTMLParser

# How much of a long posting truncate_description keeps by default.
MAX_DESCRIPTION_CHARS = 9_000

_NOISE_SELECTOR = "script, style, noscript, svg, template, iframe"
# Tags that end a line of text. Inline tags (<a>, <strong>) must not, or a
# sentence with a link in it comes out split across lines.
_BLOCK_BREAK = re.compile(r"(?i)<\s*(br|hr)\s*/?>|</\s*(p|div|li|ul|ol|h[1-6]|tr|td|th|section|article|blockquote|pre|dd|dt)\s*>")
# A description that arrived as markup rather than text: block or inline
# tags that a job posting would contain.
_HTML_DESCRIPTION = re.compile(r"<(p|div|br|li|ul|ol|h[1-6]|strong|em|b|i|span|a)\b[^>]*/?>", re.I)
_CONTROL_CHARS = re.compile(r"[\x00-\x20\x7f]")


def collapse_ws(text: str) -> str:
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def html_to_text(html: str) -> str:
    """Visible text of an HTML fragment. Block elements end a line, inline
    ones do not."""
    tree = HTMLParser(_BLOCK_BREAK.sub("\n", html))
    for node in tree.css(_NOISE_SELECTOR):
        node.decompose()
    root = tree.body or tree.root
    if root is None:
        return ""
    # No separator: the source's own spaces stay, inline tags do not split
    # words ("build <strong>fast</strong>." stays "build fast."), and the
    # newlines inserted above end the blocks.
    return collapse_ws(root.text(separator="", strip=False))


def looks_like_html(text: str) -> bool:
    return bool(_HTML_DESCRIPTION.search(text or ""))


def plain_description(text: str) -> str:
    """A stored ``scraped_jobs.description`` as readable text. A description
    stored as HTML is converted like a page fragment (block tags end a line,
    inline tags do not, entities decoded); plain text comes back unchanged."""
    if not text or not looks_like_html(text):
        return text or ""
    # &nbsp; decodes to U+00A0, which reads as a space but is not one.
    return html_to_text(text).replace("\xa0", " ")


def truncate_description(text: str, limit: int = MAX_DESCRIPTION_CHARS) -> str:
    """The head of a long posting, cut at a paragraph or line boundary."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    cut = max(head.rfind("\n\n"), head.rfind("\n"))
    return (head[:cut] if cut > limit // 2 else head).rstrip() + "\n[…posting truncated]"


def safe_http_url(url: object) -> str:
    """``url`` when it is an absolute http(s) link, else ``""``. Whitespace and
    ASCII control characters are removed first (a browser ignores a tab or
    newline inside a scheme, so ``java<TAB>script:`` must not slip through) and
    the scheme is compared case-insensitively. The one gate for every URL that
    is rendered into an ``href``."""
    if not isinstance(url, str):
        return ""
    cleaned = _CONTROL_CHARS.sub("", url)
    try:
        parts = urlsplit(cleaned)
    except ValueError:
        return ""
    if parts.scheme.lower() not in ("http", "https") or not parts.netloc or not parts.hostname:
        return ""
    return cleaned


# Tool-ish tokens: acronyms (AWS), inner capitals (PostgreSQL), digits or
# symbols (EC2, C++, CI/CD, Node.js) or plain capitalised words.
_TOKEN = re.compile(r"(?<![\w+#])[A-Za-z][A-Za-z0-9+#]*(?:[./-][A-Za-z0-9+#]+)*(?![\w+#])")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# Resume section words a rewrite may add or re-case as headings.
_HEADING_WORDS = {"summary", "profile", "experience", "work", "professional", "skills", "relevant", "technical",
                  "education", "projects", "highlights", "selected", "achievements", "employment", "history",
                  "contact", "core", "key", "competencies", "and", "of", "the", "tools", "technologies"}


def _words(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def _toolish(token: str, *, sentence_start: bool) -> bool:
    if any(c.isdigit() or c in "+#./" for c in token):
        return True
    if len(token) >= 2 and token.isupper():
        return True
    if any(c.isupper() for c in token[1:]):
        return True
    # A capital mid-sentence suggests a name (Certified, Kubernetes); at the
    # start of a line or sentence it is just grammar.
    return token[0].isupper() and not sentence_start


def unsupported_claims(source: str, output: str) -> list[str]:
    """Numbers/years and tool-like names in ``output`` that ``source`` never
    mentions (compared case-insensitively). Cheap and deliberately
    over-eager: a false alarm costs the user a glance, a missed invented
    certification costs them an interview."""
    src_words = _words(source)
    src_numbers = {n.replace(",", "") for n in _NUMBER.findall(source)}
    seen: dict[str, str] = {}
    for n in _NUMBER.findall(output):
        if n.replace(",", "") not in src_numbers:
            seen.setdefault(n, f"The number “{n}” is not in your resume.")
    for line in output.splitlines():
        for m in _TOKEN.finditer(line):
            tok = m.group(0).rstrip(".")
            if not tok or tok.lower() in src_words or tok.lower() in _HEADING_WORDS:
                continue
            # "AWS-based": a source word joined to plain lower-case words.
            parts = tok.split("-")
            if len(parts) > 1 and all(p.lower() in src_words or p.islower() for p in parts):
                continue
            before = line[:m.start()].rstrip()
            start = not before.strip("-•*·–— \t") or before.endswith((".", ":", "!", "?", "|"))
            if _toolish(tok, sentence_start=start):
                seen.setdefault(tok.lower(), f"“{tok}” is not in your resume.")
    return list(seen.values())
